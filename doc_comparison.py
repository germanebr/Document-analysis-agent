import pandas as pd
import openpyxl
import io
import random
import pickle
import string
import json
import requests
from datetime import datetime

from langchain_google_vertexai import VertexAI, HarmBlockThreshold, HarmCategory
from langchain_core.messages import HumanMessage
from google.oauth2 import service_account

from config import Config
from doc_comparison_agent_graph import DocCompAgent
from gcp_functions import run_BQ_query
from vector_space import VectorSpace

class DocumentComparator:
    def __init__(self,
                 user_id: str,
                 paths: list[str],
                 docIds: list[str] = None):
        # User ID needs to be generated when user enters the doc comparison window
        self.user_id = user_id
        self.date = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        # Process the documents and upload to the vector space
        self.paths = paths
        self.filenames = [path.split("/")[-1].rsplit(".", 1)[0] for path in self.paths]
        print(f"Got these paths:\n{self.paths}")
        print(f"Got these filenames:\n{self.filenames}\n")

        if docIds is None:
            self.doc_ids = self._get_docIds()
        else:
            self.doc_ids = docIds

        # GCP credentials
        self.credentials = service_account.Credentials.from_service_account_info(json.loads(Config.GCP_CREDENTIALS))

    def return_document_id(self):
        file_details = []
        for id, file in zip(self.doc_ids, self.filenames):
            file_details.append({"file_name": file, "id": id})

        return file_details

    def _get_docIds(self) -> list[str]:
        """Extracts the filename of each uploaded document.
        Inputs:
            - none
        Returns:
            - docIds: the list with the filenames"""
        try:
            a=[str(i) + '_' + ''.join(random.choices(string.ascii_letters + string.digits,
                                       k=8))
                for i in range(len(self.filenames))]
            
        except Exception as e:
            print("Error")    
        return [str(i) + '_' + ''.join(random.choices(string.ascii_letters + string.digits,
                                       k=8))
                for i in range(len(self.filenames))]

    def update_vector_store(self,
                            data: list) -> None:
        """Extracts the content of the given documents and uploads them into the BigQuery vector store.
        Inputs:
            - data: the raw data of each document
        Returns:
            - none"""
        
        vs = VectorSpace(paths=self.paths,
                         docs=data)

        try:
            upload_status = vs.process_docs(doc_id=self.doc_ids,
                                            cont_user_id=self.user_id)

            if not upload_status["status"]:
                raise Exception(f"Couldn't process the following documents: {upload_status['failed']}.")

        except Exception as e:
            raise e

    def _init_gemini(self) -> VertexAI:
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

        llm = VertexAI(credentials=self.credentials,
                       max_output_tokens=Config.GEMINI_MAX_OUTPUT_TOKENS,
                       model_name=Config.GEMINI_LLM_VERSION,
                       project=Config.GCP_DEV_PROJECT_ID,
                       safety_settings=safety_settings,
                       temperature=Config.GEMINI_TEMPERATURE,
                       top_p=Config.GEMINI_TOP_P,
                       verbose=False)

        return llm

    def _get_chat_history(self) -> list:
        """Retrieves the chat history from the user's Cloud Storage folder or creates a new one.
        Please modify to the corresponding code according to documentation or the team's needs.
        Inputs:
            - none
        Returns:
            - chat_history: the list with the chat history of the user"""
        
        url = "microservice url for connecting to GCP BigQuery"

        chat_history_filename = f"{self.date}_chat_history.pkl"
 
        payload = {"bucketname": "gcp-bq-bucket-name",
                   "filepath": "/".join(["cloud-storage-folder-name", self.user_id, "chat", chat_history_filename])}
        
        try:
            response = requests.post(url, json=payload)
            chat_history = pickle.loads(bytes(bytearray(response.json()["filedata"][0]["data"])))
        except:
            chat_history = []

        return chat_history

    def _update_chat_history(self,
                             prompt: str,
                             ans_meta: dict,
                             ai_response: str) -> None:
        """Updates the user's chat history with the new AI result.
        Inputs:
            - prompt: the executed prompt or user query
            - ans_meta: the answer metadata
            - ai_response: the generated answer
        Returns:
            - none"""

        # Update the chat history
        chat_history = self._get_chat_history()
        chat_history.extend([HumanMessage(content=prompt,
                                          response_metadata=ans_meta),
                             ai_response])
        # Create the pickle file
        chat_history_pickle = pickle.dumps(chat_history)

        # Upload the new chat_history
        # Please modify to the corresponding code according to documentation or the team's needs.
        url = "microservice url for uploading file from Cloud Storage"

        file_name = f"{self.date}_chat_history.pkl"
        tgt_path = "/".join(["cloud-storage-folder-name", self.user_id, "chat", file_name])
        response = requests.post(url,
                                 files={"file": (file_name, chat_history_pickle, "multipart/form-data")},
                                 data={"filepath": tgt_path, "bucketname": Config.GCP_CS_BUCKET},
                                 verify=False)
        if response.status_code == 200:
            return "/".join(["gs:/", Config.GCP_CS_BUCKET, tgt_path])
        else:
            raise Exception("File Upload Error")

    def _get_BQ_data(self,
                     feature: str) -> list:
        """Extract the Vector Store records from BigQuery based on the prompt to be executed.
        Inputs:
            - feature: the feature name of the selected prompt to run
        Returns:
            - data: the BigQuery records that resulted from the selected query"""

        # print("Getting the query for the excel files...")

        if feature == "Evaluation of Newly Added Contracts":
            query = Config.SQL_DIFFERENT_CONTRACTS.format(Config.GCP_DEV_PROJECT_ID,
                                                          Config.GCP_BQ_DATASET,
                                                          Config.GCP_BQ_EMBEDS_TABLE,
                                                          self.user_id,
                                                          self.doc_ids[1],
                                                          Config.GCP_DEV_PROJECT_ID,
                                                          Config.GCP_BQ_DATASET,
                                                          Config.GCP_BQ_EMBEDS_TABLE,
                                                          self.user_id,
                                                          self.doc_ids[0])

        elif feature == "Evaluation of Terminated Contracts":
            query = Config.SQL_DIFFERENT_CONTRACTS.format(Config.GCP_DEV_PROJECT_ID,
                                                          Config.GCP_BQ_DATASET,
                                                          Config.GCP_BQ_EMBEDS_TABLE,
                                                          self.user_id,
                                                          self.doc_ids[0],
                                                          Config.GCP_DEV_PROJECT_ID,
                                                          Config.GCP_BQ_DATASET,
                                                          Config.GCP_BQ_EMBEDS_TABLE,
                                                          self.user_id,
                                                          self.doc_ids[1])

        else:
            query = Config.SQL_COMMON_CONTRACTS.format(Config.GCP_DEV_PROJECT_ID,
                                                       Config.GCP_BQ_DATASET,
                                                       Config.GCP_BQ_EMBEDS_TABLE,
                                                       self.user_id,
                                                       self.doc_ids[0],
                                                       Config.GCP_DEV_PROJECT_ID,
                                                       Config.GCP_BQ_DATASET,
                                                       Config.GCP_BQ_EMBEDS_TABLE,
                                                       self.user_id,
                                                       self.doc_ids[1])
        # print("Query\n", query)
        return run_BQ_query(query)

    def run_prompt(self,
                   prompt: dict) -> dict:
        """Runs the user query on the LLM based on the uploaded documents.
        Inputs:
            - prompt: the prompt (with its respective feature name) to be executed
        Returns:
            - output: the dictionary with the GenAI response and the references of both documents for explainability"""

        # Determine if RAG will be used or not
        # We'll use the agent if the feature is not on the SQL_CSV_PROMPTS list
        sql_use = prompt["feature"] in Config.SQL_CSV_PROMPTS

        # Prepare the input based on using RAG or not
        if sql_use:
            print("Working with excel files")
            rag_chain = self._init_gemini()
            bq_data = self._get_BQ_data(prompt["feature"])
            context_input = "\n\nChat history:\n".join([str(bq_data),
                                                        str(self._get_chat_history())])
            genai_input = Config.GCP_LLM_SQL_PROMPT + "\n\nContext:\n".join([prompt["query"], context_input])
            genai_response = rag_chain.invoke(genai_input)

        else:
            print("Working with unstructured documents")
            rag_chain = DocCompAgent(self.doc_ids)
            genai_response = rag_chain.run_query(prompt["query"])
            # print(f"Got this response:\n{genai_response}")
            # Rename the references by the document ID
            new_refs = {}
            try:
                for id,name in zip(self.doc_ids, self.filenames):
                    new_refs[id] = genai_response["references"][name]
            except Exception as e:
                print(f"Got this error making the references:\n{e}")
                # genai_response["references"].pop(name)
            # print(f"\nThese are the references:\n{new_refs}")
            genai_response["references"] = new_refs

        # print(f"Got this:\n{genai_response}")

        # Generate a response through GenAI
        if sql_use:
            ai_ans = genai_response
        else:
            ai_ans = genai_response["answer"]

        # print(f"Got this answer:\n{ai_ans}")

        # Format the references and rearrange by page number
        references = {id: [] for id in self.doc_ids}

        if sql_use:
            _ = [references[doc["doc_id"]].append([doc[i]
                                                   for i in ["page_row", "section"]])
                 for doc in bq_data]
            
            for doc_id in references.keys():
                ref_list = [x for x in sorted(references[doc_id], key=lambda x: x[0])]
                references[doc_id] = [f"{meta[1]} - page_row {meta[0]}" for meta in ref_list]

        else:
            for key, value in genai_response["references"].items():
                references[key] = value
            
        references["feature"] = prompt["feature"]
        # print(f"Got these references:\n{references}")

        # Update chat history
        self._update_chat_history(prompt=prompt["query"],
                                  ans_meta=references,
                                  ai_response=ai_ans)

        # Prepare the output of the prompt to visualize in UI
        output = {"status": bool(ai_ans),
                  "genai_response": ai_ans,
                  "references": references}
        # print(f"This is the final output:\n{output}")
        return output

def read_excel_file(data: bytes, extension: str) -> pd.DataFrame:
    """Reads the given excel file, detects the headers row, and ignores any data above that.
    Inputs:
        - data: the content of the excel file in bytes
        - extension: the type of tabular document
    Returns:
        - df: the pandas dataframe with all the content of the file"""

    # Read the file
    if extension == "csv":
        # print("It's a csv")
        df = pd.read_csv(io.BytesIO(data), header=None , sep=",", encoding="latin-1")
    else:
        # print("It's an excel")
        df = pd.read_excel(io.BytesIO(data), header=None)

        # Detect the header row index
        headers_idx = df.apply(lambda row: row.notnull().any(), axis=1).idxmax()

        # Set the header row for the dataframe
        df.columns = df.iloc[headers_idx]

        # Load the workbook using openpyxl to read hyperlink data
        wb = openpyxl.load_workbook(io.BytesIO(data))
        ws = wb.active

        # Iterate through the DataFrame and extract hyperlinks
        for i, row in df.iterrows():
            for j, _ in enumerate(row):
                cell = ws.cell(row=i + 2, column=j + 1)  # +2 for header row adjustment
                if cell.hyperlink:
                    df.at[i, df.columns[j]] = cell.hyperlink.target

        # Ignore any data above the headers row
        df = df.iloc[headers_idx + 1:]
    # print(df)
    return df


def get_from_cloud_storage(user: str, filename: str):
    """Retrieves the data of the selected file from Google Cloud Storage
    Please modify to the corresponding code according to documentation or the team's needs.
    Inputs:
        - user: the username to look for their folder on GCS
        - filename: the filename with extension of the desired document
    Returns:
        - data: the extracted data of the document (bytes for non-tabular files and a pandas dataframe for tabular files)"""

    url = "microservice url for connecting to GCP Cloud Storage"
 
    payload = {"bucketname": "gcp-bucket-name",
               "filepath": "/".join(["cloud-storage-folder-name", user, "docs", filename])}
    
    response = requests.post(url, json=payload)
    file_content = bytes(bytearray(response.json()["filedata"][0]["data"]))

    extension = filename.rsplit(".", 1)[-1]

    if extension in ["xlsx", "csv"]:
        # print("Got an excel or csv")
        data = read_excel_file(file_content, extension)

    else:
        data = file_content

    return data

# if __name__ == "__main__":
# Simulate the user ID (needs to be generated when the user enters into the app)
# user_id = "test"
# Get the paths of the files to upload
# ---------------------------- PSMF TEST ----------------------------
# print("------------------------ PSMF RELEVANCE TEST ------------------------\n\n")
# file2_path = "sample_excel_1.xlsx"
# file3_path = "sample_excel_2.xlsx"

# paths = [file2_path, file3_path]
# print("Reading the documents from Cloud Storage")
# docs_data = [get_from_cloud_storage(user_id, path) for path in paths]
# print("Got the paths and raw data")

# Update the vector space when user presses the upload button
#     agent = DocumentComparator(user_id=user_id,
#                                paths=paths)

#     print("Uploading the documents")
#     agent.update_vector_store(docs_data)
#     print("Done!")

#     print("\nRunning prompts\n")
#     prompt1 = {"feature": "Evaluation of Common Contracts",
#                "query": """You are a Pharmacovigilance contract professional who needs to evaluate the content of multiple commercial agreements. You will receive a list with all the common contracts that are between two documents.
# Generate a paragraph explaining all the details of the given list.
# Make sure to mention information of every contract ID that is in there."""}
#     prompt2 = {"feature": "Evaluation of Terminated Contracts",
#                "query": """You are a Pharmacovigilance contract professional who needs to evaluate the content of multiple commercial agreements. You will receive a list with all the contracts that were terminated.
# Generate a paragraph explaining all the details of the given list.
# Make sure to mention information of every contract ID that is in there."""}
#     prompt3 = {"feature": "Evaluation of Newly Added Contracts",
#                "query": """You are a Pharmacovigilance contract professional who needs to evaluate the content of multiple commercial agreements. You will receive a list with all the new contracts that were created.
# Generate a paragraph explaining all the details of the given list.
# Make sure to mention information of every contract ID that is in there."""}
#     prompts = [prompt1, prompt2, prompt3]

#     for prompt in prompts:
#         try:
#             response = agent.run_prompt(prompt)
#             # Show on the UI the prompt and the response
#             print(prompt)
#             print(response)
#             print()
#         except Exception as e:
#             raise(e)

# Generate the excel file if the user presses the 'Export results' button
# try:
#     agent.export_results()
#     print("Exported the results")
# except Exception as e:
#     raise (e)

# ---------------------------- GLOBAL/LOCAL TEST ----------------------------
# print("\n\n------------------------ GLOBAL/LOCAL TEST ------------------------\n\n")
# file4_path = "Template Copy.pdf"
# file5_path = "Contract Copy.docx"

# paths = [file4_path, file5_path]

# print("Reading the documents from Cloud Storage")
# docs_data = [get_from_cloud_storage(user_id, path) for path in paths]
# print("Got the paths and raw data")

# Update the vector space when user presses the upload button
# agent = DocumentComparator(user_id=user_id,
#                            paths=paths)

# print("Uploading the documents")
# agent.update_vector_store(docs_data)
# print("Done!")

# print("\nRunning prompts\n")
# prompt1 = {"feature": 'Pharmacovigilance Summary',
#            "query": 'You are a Pharmacovigilance contract professional who needs to compare a partner document with a given template. Provide a summary with all the features that are consistent between the template and the contract.'}
# prompts = [prompt1]

# for prompt in prompts:
#     try:
#         response = agent.run_prompt(prompt)
#         # Show on the UI the prompt and the response
#         print(prompt)
#         print(response)
#         print()
#     except Exception as e:
#         raise(e)

# Generate the excel file if the user presses the 'Export results' button
# try:
#     agent.export_results()
#     print("Exported the results")
# except Exception as e:
#     raise (e)