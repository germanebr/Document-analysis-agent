from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory, VertexAIEmbeddings
from pydantic import BaseModel, Field
from google.oauth2 import service_account
import operator
import json
from typing import Annotated, Dict
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph

from config import Config
from gcp_functions import run_BQ_query


class State(TypedDict):
    """Base structure for storing the data of the agent's processing"""
    prompt:str = Field(description = "The user query to be answered by the agent")
    docs_data:list = Field(description = "List with the content from the uploaded documents")
    references:Annotated[list, operator.add] = Field(description = "List of references of each answer from the key features")
    classification:Dict[str, str] = Field(description = "The classification of the two uploaded documents",
                                          examples = {"template": "document_1",
                                                      "contract": "document_2"})
    answer:str = Field(description = "The final answer that answers the given user query")

class ClassificationState(BaseModel):
    """Base structure for the classifying the uploaded documents"""
    template:str = Field(description = "The filename of the document that corresponds to the business template")
    contract:str = Field(description = "The filename of the document that corresponds to the real business contract")

class DocCompAgent():
    """The agent graph that manages the overall execution of the given query.
    
    More robust than just calling directly the LLM for managing which document is the template and which the business agreement,
    as well as to have better organization on how to retrieve data from BigQuery to answer the given query."""

    def __init__(self, doc_ids):
        try:
            self.doc_ids = doc_ids
            self.credentials = service_account.Credentials.from_service_account_info(json.loads(Config.GCP_CREDENTIALS))
            self.model = self._init_gemini()
            self.embedding_model =  VertexAIEmbeddings(model_name = Config.GEMINI_EMBEDDING_MODEL,
                                                       project = Config.GCP_DEV_PROJECT_ID,
                                                       credentials = self.credentials)
            self.agent = self._compile_agent()
            print("Agent initialization completed!")
        except Exception as e:
            print(f"Error initializing the agent:\n{e}")

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

        llm = ChatVertexAI(credentials = self.credentials,
                           max_output_tokens = Config.GEMINI_MAX_OUTPUT_TOKENS,
                           model = Config.GEMINI_LLM_VERSION,
                           project = Config.GCP_DEV_PROJECT_ID,
                           safety_settings = safety_settings,
                           temperature = Config.GEMINI_TEMPERATURE,
                           top_p = Config.GEMINI_TOP_P,
                           verbose = False)
        print("LLM initialized correclty!")
        return llm
    
    def _get_docs_data(self, state:State):
        """Retrieve the content from the uplodaded documents for answering the user query."""
        print("----- Getting documents data -----")
        docs_data = run_BQ_query(Config.LANGCHAIN_SQL_GET_DOCS_DATA.format(doc_1 = self.doc_ids[0],
                                                                           doc_2 = self.doc_ids[1]))
        print("\t> Done!")
        return {"docs_data": docs_data}

    def _classify_documents(self, state:State):
        """Graph block that determines which document is the business template and which one the real agreement."""
        print("----- Classifying the documents -----")
        structrured_llm = self.model.with_structured_output(ClassificationState)

        prompt = Config.LANGCHAIN_CLASSIFY_DOCUMENTS_MSG.format(uploaded_docs = state["docs_data"])
        
        response = structrured_llm.invoke(prompt)
        print("\t> Done classifying the documents!")
        return {"classification": {"template": response.template,
                                   "contract": response.contract}}

    def _generate_answer(self, state:State):
        """Graph block that generates a final answer based on the given user query."""
        print("----- Generating answer -----")
        prompt = Config.LANGCHAIN_DOC_COMPARISON_BASE_PROMPT.format(classification = state["classification"],
                                                                    user_query = state["prompt"],
                                                                    docs_content = state["docs_data"])
        
        response = self.model.invoke(prompt)
        print(f"\t> Done answering the query!")
        return {"answer": response.content}

    def _get_references(self, state:State):
        """Graph block for selecting the references used on the generated response."""
        print(f"----- Getting the references -----")
        query_emb = self.embedding_model.embed_query(state["answer"])
        res = run_BQ_query(Config.LANGCHAIN_SQL_SIMILARITY_QUERY.format(query_emb = query_emb,
                                                                        doc_1 = self.doc_ids[0],
                                                                        doc_2 = self.doc_ids[1],
                                                                        k = Config.LANGCHAIN_RETRIEVER_NUMBER_RESULTS))
        
        references = {i["filename"]: [] for i in res}
        _ = [references[i["filename"]].append({"section": i["section"],
                                               "page_row": i["page_row"]})
             for i in res]
                
        print(f"\t> Done getting the references!")
        return {"references": [references]}
        
    def _compile_agent(self):
        """Builds the agent graph."""
        # Build the graph
        builder = StateGraph(State)

        # Define the nodes
        builder.add_node("get_docs_data", self._get_docs_data)
        builder.add_node("docs_classifier", self._classify_documents)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_node("get_references", self._get_references)

        # Define the relationships
        builder.add_edge(START, "get_docs_data")
        builder.add_edge("get_docs_data", "docs_classifier")
        builder.add_edge("docs_classifier", "generate_answer")
        builder.add_edge("generate_answer", "get_references")
        builder.add_edge("get_references", END)

        agent = builder.compile()
        # print(agent.get_graph().draw_mermaid())
        return agent

    def run_query(self, query:str):
        """Runs the given query or prompt within the agent graph."""
        print("Running agent for user query")
        res = self.agent.invoke({"prompt": query})
        return {"answer": res["answer"],
                "references": self._order_refs(res["references"])}

    def _order_refs(self, refs):
        print("Ordering references")
        """Orders the given references by document and by page."""
        references = {}
        for block in refs:
            for key in block.keys():
                references[key] = set()

        for block in refs:
            for key, values in block.items():
                for val in values:
                    references[key].add(" - page_row ".join([str(i) for i in val.values()]))

        for key, values in references.items():
            refs_list = list(values)
            ref_with_page_number = []
            for i in refs_list:
                try:
                    page_num = int(i.rsplit('page_row ', 1)[1])
                    ref_with_page_number.append((page_num, i))
                except Exception as e:
                    print(f"Could not extract page number from {i}.")

            ref_with_page_number.sort()
            references[key] = [i[1] for i in ref_with_page_number]
            
        print("References ordered correctly!")
        return references