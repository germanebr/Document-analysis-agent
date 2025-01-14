from typing import Tuple

import pandas as pd
import openpyxl
import io
import random
import pickle
import string
import requests
from datetime import datetime
from google.cloud import bigquery
from config import Config
from langchain_google_vertexai import VertexAI, HarmBlockThreshold, HarmCategory
from langchain_core.messages import HumanMessage
from google.oauth2 import service_account
from google.cloud import storage

from vector_space import VectorSpace
from gcp_functions import run_BQ_query
from document_comparison_agent_graph import DocCompAgent

class DocumentComparator:
    def __init__(self,
                 user_id: str,
                 paths: list[str],
                 docIds: list[str]=None):
        # User ID needs to be generated when user enters the doc comparison window
        self.user_id = user_id

        self.date = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

        # Process the documents and upload to the vector space
        self.paths = paths
        self.filenames = [path.split("/")[-1].rsplit(".", 1)[0] for path in self.paths]
        if docIds is None:
            self.doc_ids = self._get_docIds()
        else:
            self.doc_ids = docIds

        # GCP credentials
        self.credentials = service_account.Credentials.from_service_account_file(Config.GCP_CREDENTIALS)

    def return_document_id(self):
        return self.doc_ids

    def _get_docIds(self) -> list[str]:
        """Extracts the filename of each uploaded document.
        Inputs:
            - none
        Returns:
            - docIds: the list with the filenames"""

        # Please change to whatever format you need to distinguish each document
        return [''.join(random.choices(string.ascii_letters + string.digits,
                                       k=8))
                for _ in range(len(self.filenames))]

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
        Inputs:
            - none
        Returns:
            - chat_history: the list with the chat history of the user"""

        storage_client = storage.Client(credentials=self.credentials)
        bkt = storage_client.bucket(Config.GCP_CS_BUCKET)

        chat_history_filename = f"{self.date}_chat_history.pkl"

        try:
            blob = bkt.blob("/".join(["doc_comparison", self.user_id, "chat", chat_history_filename]))

            with blob.open("rb") as f:
                chat_history = pickle.load(f)

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
        # For general testing, modify this function to the according GCP Documentation
        url = "{cloud_storage_connection_microservice_url}"

        file_name = f"{self.date}_chat_history.pkl"
        tgt_path = "/".join(["doc_comparison", self.user_id, "chat", file_name])
        response = requests.post(url,
                                 files={"file": (file_name, chat_history_pickle, "multipart/form-data")},
                                 data={"filepath": tgt_path, "bucketname": Config.GCP_CS_BUCKET},
                                 verify=False)
        if response.status_code == 200:
            return "/".join(["gs:/", Config.GCP_CS_BUCKET, tgt_path])
        else:
            print("Error in File Upload to GCS")
            print(f"Status Code: {response.status_code}")
            print(f"Status Text: {response.text}")
            raise Exception("File Upload Error")        
        
    def _get_BQ_data(self,
                     feature:str) -> list:
        """Extract the Vector Store records from BigQuery based on the prompt to be executed.
        Inputs:
            - feature: the feature name of the selected prompt to run
        Returns:
            - data: the BigQuery records that resulted from the selected query"""
        
        print("Getting the query for the excel files...")
        
        if feature == "Evaluation of Newly Added Contracts":
            query = Config.SQL_PSMF_DIFFERENT_CONTRACTS.format(Config.GCP_DEV_PROJECT_ID,
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
            query = Config.SQL_PSMF_DIFFERENT_CONTRACTS.format(Config.GCP_DEV_PROJECT_ID,
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
            query = Config.SQL_PSMF_COMMON_CONTRACTS.format(Config.GCP_DEV_PROJECT_ID,
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
        # We'll use RAG if the feature is not on the SQL_VST_PROMPTS list
        sql_use = prompt["feature"] in Config.SQL_VST_PROMPTS

        # Prepare the input based on using RAG or not
        if sql_use:
            rag_chain = self._init_gemini()
            # print("\nBQ Data:\n", self._get_BQ_data(prompt["feature"]))
            # print("Chat history:\n", self._get_chat_history())
            bq_data = self._get_BQ_data(prompt["feature"])
            context_input = "\n\nChat history:\n".join([str(bq_data),
                                                        str(self._get_chat_history())])
            # print("\nContext input:\n", context_input)
            genai_input = "\n\nContext:\n".join([prompt["query"], context_input])
            # print("Full input:\n", genai_input)
            genai_response = rag_chain.invoke(genai_input)

        else:
            rag_chain = DocCompAgent(self.doc_ids)
            genai_response = rag_chain.run_query(prompt["query"])

        # Generate a response through GenAI
        print("GenAI Response:\n", genai_response)

        if sql_use:
            ai_ans = genai_response
        else:
            ai_ans = genai_response["answer"]

        # Format the references and rearrange by page number
        references = {id: [] for id in self.doc_ids}

        if sql_use:
            # print("The BQ data:\n", bq_data)
            # for doc in genai_response:
            #     print(doc)
            _ = [references[doc["doc_id"]].append([doc[i]
                                                   for i in ["page_row", "section"]])
                 for doc in bq_data]
            # _ = [references[doc["doc_id"]].append([doc[i]
            #                                       for i in ["page_row", "section"]])
            #      for doc in genai_response]
            for doc_id in references.keys():
                ref_list = [x for x in sorted(references[doc_id], key=lambda x: x[0])]
                references[doc_id] = [f"{meta[1]} - page_row {meta[0]}" for meta in ref_list]

        else:
            for key,value in genai_response["references"].items():
                references[key] = value        

        references["feature"] = prompt["feature"]
        # print(references)

        # Update chat history
        self._update_chat_history(prompt=prompt["query"],
                                  ans_meta=references,
                                  ai_response=ai_ans)

        # Prepare the output of the prompt to visualize in UI
        output = {"status": bool(ai_ans),
                  "genai_response": ai_ans,
                  "references": references}
        print(output)
        return output

    def export_results(self, save_to_local: bool = False) -> Tuple:
        """Generates an excel file with all the executed prompts and their coresponding answers.
        Inputs:
            - none
        Returns:
            - none"""

        # Determine which is doc1 and which doc2
        df_files = pd.DataFrame(data={"Document 1": [self.filenames[0]],
                                      "Document 2": [self.filenames[1]]})

        # Process the chat history
        features = []
        prompts = []
        refs_1 = []
        refs_2 = []
        ai_res = []

        for message in self._get_chat_history():
            if isinstance(message, str):
                ai_res.append(message)
            else:
                doc_keys = [key for key in message.response_metadata.keys() if key != "feature"]
                features.append(message.response_metadata["feature"])
                refs_1.append(message.response_metadata[doc_keys[0]])
                refs_2.append(message.response_metadata[doc_keys[1]])
                prompts.append(message.content)

        df_hist = pd.DataFrame(data={"Feature": features,
                                     "Prompt": prompts,
                                     "References for Document 1": refs_1,
                                     "References for Document 2": refs_2,
                                     "AI Interpretation": ai_res})

        export_filename = f"{self.user_id}_{self.date}_Document_Comparison_Results.xlsx"
        if save_to_local:
            # Export into the excel file


            with pd.ExcelWriter(export_filename) as writer:
                df_files.to_excel(writer,
                                  sheet_name="Document list",
                                  index=False)
                df_hist.to_excel(writer,
                                 sheet_name="Doc Comparison Results",
                                 index=False)
        else:
            return df_files, df_hist, export_filename

    def clear_session(self) -> None:
        """Deletes all data related to the uploaded documents from the Vector Space
        Inputs:
            - none
        Returns:
            - none"""

        client = bigquery.Client(project=Config.GCP_DEV_PROJECT_ID,
                                 credentials=self.credentials)

        query = f"DELETE FROM `{Config.GCP_DEV_PROJECT_ID}.{Config.GCP_BQ_DATASET}.{Config.GCP_BQ_EMBEDS_TABLE}` WHERE contract_user_id = '{self.user_id}';"

        _ = [row.values() for row in client.query(query).result()]


def read_excel_file(data: bytes, extension: str) -> pd.DataFrame:
    """Reads the given excel file, detects the headers row, and ignores any data above that.
    Inputs:
        - data: the content of the excel file in bytes
        - extension: the type of tabular document
    Returns:
        - df: the pandas dataframe with all the content of the file"""

    # Read the file
    if extension == "csv":
        df = pd.read_csv(io.BytesIO(data), header=None)
    else:
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
    return df


def get_from_cloud_storage(user: str, filename: str):
    """Retrieves the data of the selected file from Google Cloud Storage
    Inputs:
#         - user: the username to look for their folder on GCS
#         - filename: the filename with extension of the desired document
#     Returns:
#         - data: the extracted data of the document (bytes for non-tabular files and a pandas dataframe for tabular files)"""

    credentials = service_account.Credentials.from_service_account_file(Config.GCP_CREDENTIALS)

    storage_client = storage.Client(credentials=credentials)
    bkt = storage_client.bucket(Config.GCP_CS_BUCKET)
    blob = bkt.blob("/".join(["doc_comparison", user, "docs", filename]))

    extension = filename.rsplit(".", 1)[-1]

    if extension in ["xlsx", "csv"]:
        file_content = blob.download_as_bytes()
        data = read_excel_file(file_content, extension)

    else:
        with blob.open("rb") as file:
            data = file.read()

    return data


def get_raw_data(path: str):
    # Get the extension
    extension = path.rsplit(".", 1)[-1]

    if extension not in ["xlsx", "csv"]:
        print("Reading non-tabular file")
        with open(path, "rb") as file:
            doc_data = file.read()
        return doc_data
    else:
        print("Reading tabular file")
        with open(path, 'rb') as file:
            excel_bytes = file.read()
        return read_excel_file(excel_bytes, extension)


if __name__ == "__main__":
    # Simulate the user ID (needs to be generated when the user enters into the app)
    user_id = "test"
    # Get the paths of the files to upload
    
    print("------------------------ EXCEL DOCUMENTS TEST ------------------------\n\n")
    file2_path = "sample_excel_1.xlsx"
    file3_path = "sample_excel_2.xlsx"

    paths = [file2_path, file3_path]
    print("Reading the documents from Cloud Storage")
    docs_data = [get_from_cloud_storage(user_id, path) for path in paths]
    print("Got the paths and raw data")

    # Update the vector space when user presses the upload button
    agent = DocumentComparator(user_id=user_id,
                               paths=paths)
    
    print("Uploading the documents")
    agent.update_vector_store(docs_data)
    print("Done!")
    
    print("\nRunning prompts\n")
    prompt1 = {"feature": "Evaluation of Common Contracts",
               "query": """You are a Pharmacovigilance contract professional who needs to evaluate the content of multiple commercial agreements. You will receive a list with all the common contracts that are between two documents.
Generate a paragraph explaining all the details of the given list.
Make sure to mention information of every contract ID that is in there."""}
    prompt2 = {"feature": "Evaluation of Terminated Contracts",
               "query": """You are a Pharmacovigilance contract professional who needs to evaluate the content of multiple commercial agreements. You will receive a list with all the contracts that were terminated.
Generate a paragraph explaining all the details of the given list.
Make sure to mention information of every contract ID that is in there."""}
    prompt3 = {"feature": "Evaluation of Newly Added Contracts",
               "query": """You are a Pharmacovigilance contract professional who needs to evaluate the content of multiple commercial agreements. You will receive a list with all the new contracts that were created.
Generate a paragraph explaining all the details of the given list.
Make sure to mention information of every contract ID that is in there."""}
    prompts = [prompt1, prompt2, prompt3]

    for prompt in prompts:
        try:
            response = agent.run_prompt(prompt)
            # Show on the UI the prompt and the response
            print(prompt)
            print(response)
            print()
        except Exception as e:
            raise(e)

    # Generate the excel file if the user presses the 'Export results' button
    try:
        agent.export_results()
        print("Exported the results")
    except Exception as e:
        raise (e)

    print("\n\n------------------------ CONTRACT PDF TEST ------------------------\n\n")
    file4_path = "pdf_copy_1.pdf"
    file5_path = "pdf_copy_2.docx"

    paths = [file4_path, file5_path]

    print("Reading the documents from Cloud Storage")
    docs_data = [get_from_cloud_storage(user_id, path) for path in paths]
    print("Got the paths and raw data")

    # Update the vector space when user presses the upload button
    agent = DocumentComparator(user_id=user_id,
                               paths=paths)
    
    print("Uploading the documents")
    agent.update_vector_store(docs_data)
    print("Done!")

    print("\nRunning prompts\n")
    prompt1 = {"feature": 'Pharmacovigilance Data with Product Quality Complaints',
               "query": '''You are a Pharmacovigilance contract professional who needs to compare a partner document with a template.
               Specifically address the following:
               1. **Does the partner document describe procedures for managing the interface between safety information and product quality complaints?**
               If so, describe these procedures and compare them to the template's requirements.  Highlight any discrepancies.
               2. **Does the partner document specify the exchange of lot/batch numbers and expiration dates (when available) in cases where
               Pharmacovigilance Data includes product quality complaints?**
               If so, describe the process and compare it to the template's requirements. Highlight any discrepancies.
               3. **Does the partner document reference a separate Quality Agreement for handling product quality complaints?**
               If so, describe how this is addressed and compare it to the template's requirements. Highlight any discrepancies.
               4. **Provide a summary table outlining the key similarities and differences between the template and the partner document regarding the handling of
               Pharmacovigilance Data with Product Quality Complaints.**
               Provide your analysis in a clear and concise manner. If information is missing from the partner document, clearly state this.'''}
    prompts = [prompt1]

    for prompt in prompts:
        try:
            response = agent.run_prompt(prompt)
            # Show on the UI the prompt and the response
            print(prompt)
            print(response)
            print()
        except Exception as e:
            raise(e)

    # Generate the excel file if the user presses the 'Export results' button
    try:
        agent.export_results()
        print("Exported the results")
    except Exception as e:
        raise (e)

    # Clear the vector space everytime the user logs out of the application or uploads new documents to compare
    # agent.clear_session()