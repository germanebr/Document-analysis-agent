from langchain_google_vertexai import VertexAI, HarmBlockThreshold, HarmCategory, VertexAIEmbeddings
from google.oauth2 import service_account
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from config import Config
from gcp_functions import run_BQ_query

class State(TypedDict):
    """Base structure for storing the data of the agent's processing"""
    prompt: str
    key_features: list # List to store the results of each key feature extracted from the main user_prompt
    references: Annotated[list, operator.add] # List of references of each answer from the key features
    classification: str
    answer: str

class ReferenceState(TypedDict):
    """Base structure for the processing of references inside the agent"""
    feature: str

class DocCompAgent():
    """The agent graph that manages the overall execution of the given query.
    
    More robust than just calling directly the LLM for managing which document is the template and which the business agreement,
    as well as to have better organization on how to retrieve data from BigQuery to answer the given query."""

    def __init__(self, doc_ids):
        self.doc_ids = doc_ids
        self.credentials = service_account.Credentials.from_service_account_file(Config.GCP_CREDENTIALS)
        self.model = self._init_gemini()
        self.embedding_model =  VertexAIEmbeddings(model_name = Config.GEMINI_EMBEDDING_MODEL,
                                                   project = Config.GCP_DEV_PROJECT_ID,
                                                   credentials = self.credentials)
        self.agent = self._compile_agent()

    def _init_gemini(self):
        """Initializes the base LLM for the prompt execution.
        Inputs:
            - none
        Returns:
            - llm: the initialized LLM model with all its configuration"""

        safety_settings = {HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                           HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                           HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                           HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                           HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE}

        llm = VertexAI(credentials = self.credentials,
                       max_output_tokens = Config.GEMINI_MAX_OUTPUT_TOKENS,
                       model_name = Config.GEMINI_LLM_VERSION,
                       project = Config.GCP_DEV_PROJECT_ID,
                       safety_settings = safety_settings,
                       temperature = Config.GEMINI_TEMPERATURE,
                       top_p = Config.GEMINI_TOP_P,
                       verbose = False)

        return llm

    def _generate_key_features(self, state:State):
        """Graph block for determining key features to solve from the given prompt."""
        print("----- Getting keywords -----")
        features = self.model.invoke(Config.LANGCHAIN_KEYWORD_EXTRACTOR_MSG.format(query = state["prompt"]))
        print(features, "\n")
        return {"key_features": [i for i in features.split("\n") if i]}

    def _continue_to_references(self, state:State):
        """Function that manages parallel execution for retrieving the BigQuery data from each generated key feature."""
        return [Send("BQ_retriever", {"feature": feature}) for feature in state["key_features"]]

    def _get_references(self, state:ReferenceState):
        """Graph block for connecting to BigQuery and retrieving the most relevant information based on the given key feature or prompt."""
        print(f"----- Getting the references for {state["feature"]} -----")
        emb = self.embedding_model.embed_query(state["feature"])    
        
        res = run_BQ_query(Config.LANGCHAIN_SQL_SIMILARITY_QUERY.format(query_emb = emb,
                                                                        doc_1 = self.doc_ids[0],
                                                                        doc_2 = self.doc_ids[1],
                                                                        k = Config.LANGCHAIN_RETRIEVER_NUMBER_RESULTS))
        
        references = {i["filename"]: [] for i in res}
        _ = [references[i["filename"]].append({"section": i["section"],
                                               "page_row": i["page_row"]})#,
                                               #"content": i["content"]})
             for i in res]
                
        print(f"\t{state["feature"]}: {references}\n")
        return {"references": [references]}

    def _classify_documents(self, state:State):
        """Graph block that determines which document is the business template and which one the real agreement."""
        print("----- Classifying the documents -----")
        references = {}

        for reference in state["references"]:
            for doc_name, metadata in reference.items():
                if doc_name not in references:
                    references[doc_name] = []
                
                references[doc_name].extend(metadata)

        prompt = Config.LANGCHAIN_CLASSIFY_DOCUMENTS_MSG.format(references = references)
        
        response = self.model.invoke(prompt)
        print(f"Classification: {response}\n")
        return {"classification": response}

    def _generate_answer(self, state:State):
        """Graph block that generates a final answer based on the prevoius blocks information."""
        print("----- Generating answer -----")
        features = "\n".join(state["key_features"])
        references = "\n".join(str(state["references"]))

        prompt = Config.LANGCHAIN_DOC_COMPARISON_BASE_PROMPT.format(classification = state["classification"],
                                                                    user_query = state["prompt"],
                                                                    key_features = features,
                                                                    references = references)
        
        response = self.model.invoke(prompt)
        print(response)
        return {"answer": response}
    
    def _compile_agent(self):
        """Builds the agent graph."""
        # Build the graph
        builder = StateGraph(State)

        # Define the nodes
        builder.add_node("generate_key_features", self._generate_key_features)
        builder.add_node("BQ_retriever", self._get_references)
        builder.add_node("docs_classifier", self._classify_documents)
        builder.add_node("generate_answer", self._generate_answer)

        # Define the relationships
        builder.add_edge(START, "generate_key_features")
        builder.add_conditional_edges("generate_key_features", self._continue_to_references, ["BQ_retriever"])
        builder.add_edge("BQ_retriever", "docs_classifier")
        builder.add_edge("docs_classifier", "generate_answer")
        builder.add_edge("generate_answer", END)

        agent = builder.compile()
        print(agent.get_graph().draw_mermaid())
        return agent

    def run_query(self, query:str):
        """Runs the given query or prompt within the agent graph."""
        agent_flow = self.agent.invoke({"prompt": query})
        return {"answer": agent_flow["answer"],
                "references": self._order_refs(agent_flow["references"])}
    
    def _order_refs(self, refs):
        """Orders the given references by document and by page."""
        references = {}
        for block in refs:
            for key in block.keys():
                references[key] = []

        for block in refs:
            for key,values in block.items():
                references[key] += values

        for key,values in references.items():
            references[key] = sorted(values, key=lambda item: item["page_row"])

        return references
    
# if __name__ == "__main__":
#     doc_ids = ["PrSKLU0R", "WMjPsSDZ"]
#     query = """You are a Pharmacovigilance contract professional who needs to compare a partner document with the JNJ template.  The JNJ template section concerning Pharmacovigilance Data and Product Quality Complaints specifies the following:

# * **3.7 Interface Management:**  Both parties must have procedures to manage the interaction between safety information and product quality complaints.  When Pharmacovigilance Data includes product quality complaints, lot/batch numbers and expiration dates (when available) must be exchanged according to the timelines, format, and method specified in Schedule 5.

# * **3.8 Quality Agreement Adherence:**  The handling of product quality complaints will adhere to a separate Quality Agreement, where applicable.


# Compare the uploaded partner document against these requirements.  Specifically address the following:

# 1. **Does the partner document describe procedures for managing the interface between safety information and product quality complaints?** If so, describe these procedures and compare them to the JNJ template''s requirements.  Highlight any discrepancies.

# 2. **Does the partner document specify the exchange of lot/batch numbers and expiration dates (when available) in cases where Pharmacovigilance Data includes product quality complaints?** If so, describe the process and compare it to the JNJ template''s requirements (including adherence to Schedule 5, if applicable). Highlight any discrepancies.

# 3. **Does the partner document reference a separate Quality Agreement for handling product quality complaints?** If so, describe how this is addressed and compare it to the JNJ template''s requirements. Highlight any discrepancies.

# 4. **Provide a summary table outlining the key similarities and differences between the JNJ template and the partner document regarding the handling of Pharmacovigilance Data with Product Quality Complaints.**

# Provide your analysis in a clear and concise manner. If information is missing from the partner document, clearly state this."""

#     agent = DocCompAgent(doc_ids)
#     response = agent.run_query(query)
#     print(response)