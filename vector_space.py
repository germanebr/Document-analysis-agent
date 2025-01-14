import io
import pandas as pd
import openpyxl
from typing import List
from azure_doc_int import AzureDocIntParser
from google.cloud import bigquery
from google.oauth2 import service_account
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore
from config import Config

class VectorSpace():
    def __init__(self,
                 paths:List[str],
                 docs:List,
                 chunk_size:int = 8000,
                 overlap:float = 800) -> None:
        
        self.chunk_size = chunk_size
        self.credentials = service_account.Credentials.from_service_account_file(Config.GCP_CREDENTIALS)
        self.overlap = overlap
        self.paths = paths
        self.docs = docs
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                                            chunk_overlap = overlap,
                                                            separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""])
    
    def create_BQ_table(self,
                        table_id:str) -> None:
        """Creates BigQuery table with table_id provided, or if table already exists, does nothing.
        Inputs:
            - table_id: the GCP uri for creating the table
        Returns:
            - none"""

        client = bigquery.Client(project = Config.GCP_DEV_PROJECT_ID,
                                 credentials = self.credentials)

        schema = [bigquery.SchemaField("doc_id", "STRING", mode = "REQUIRED"),
                  bigquery.SchemaField("section", "STRING", mode = "REQUIRED"),
                  bigquery.SchemaField("filename", "STRING", mode = "NULLABLE"),
                  bigquery.SchemaField("contract_user_id", "STRING", mode = "REQUIRED"),
                  bigquery.SchemaField("page_row", "INTEGER", mode = "REQUIRED"),
                  bigquery.SchemaField("chunk", "INTEGER", mode = "REQUIRED"),
                  bigquery.SchemaField("content", "STRING", mode = "REQUIRED"),
                  bigquery.SchemaField("embedding", "FLOAT", mode = "REPEATED")]

        #Create the table
        table = bigquery.Table(table_id,
                               schema = schema)
        try:
            table = client.create_table(table)  # Make an API request.
            # print("\t> Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id))

        except:
            # print("\t> Table already created")
            pass

    def run_BQ_query(self,
                     query:str) -> List:
        """Runs the given SQL query in BigQuery.
        Inputs:
            - query: the BigQuery query to execute
        Returns:
            - res: the response obtained"""

        client = bigquery.Client(project = Config.GCP_DEV_PROJECT_ID,
                                 credentials = self.credentials)

        res = [row.values() for row in client.query(query).result()]
        return res
    
    def create_text_chunks(self,
                           doc_json:dict,
                           doc_id:str,
                           cont_user_id:str):
        """Stores the data and its embeddings from the document json into the given BigQuery table for the RAG vector space
        Inputs:
            - doc_json: the data of the document retrieved with Azure Document Intelligence including text and tables
            - doc_id: the ICD document ID
            - cont_user_id: the ID of the ICD contract (for PV Assessment) OR the user (for document comparison)
        Returns:
            - chunks: the list with the chunks of the document
            - metadata: the list with the corresponding metadata of each chunk"""
        
        # Get the chunks of the main text
        chunks = []
        metadata = []

        # Divide into chunks the main text
        for page,sections in doc_json["doc_text"].items():
            page_num = int(page.split("_")[-1])

            for section,content in sections.items():
                section_chunks = self.text_splitter.split_text("\n\n".join(content))

                for chunk in section_chunks:
                    meta = {"doc_id": doc_id,
                            "contract_user_id": cont_user_id,
                            "filename": doc_json["filename"],
                            "section": section,
                            "page_row": page_num,
                            "chunk": len(chunks)}
                    chunks.append(chunk)
                    metadata.append(meta)

        # Include the content of the tables into the chunks
        for table,values in doc_json["doc_tables"].items():
            chunks.append(values["content"])
            metadata.append({"doc_id": doc_id,
                             "contract_user_id": cont_user_id,
                             "filename": doc_json["filename"],
                             "section": table,
                             "page_row": values["page"],
                             "chunk": len(chunks)})
            
        return chunks, metadata

    def store_vector_data_BQ(self,
                             chunks:list[str],
                             metadata:list[dict],) -> None:
        """Stores the documents' chunks and its corresponding metadata into the BigQuery table to create/update the RAG vector space
        Inputs:
            - chunks: the list with the chunks to upload
            - metadata: the list with the metadata of every given chunk
        Returns:
            - none"""
        
        embedding_model = VertexAIEmbeddings(model_name = Config.GEMINI_EMBEDDING_MODEL,
                                             project = Config.GCP_DEV_PROJECT_ID,
                                             credentials = self.credentials)
        
        bq_store = BigQueryVectorStore(project_id = Config.GCP_DEV_PROJECT_ID,
                                       location = Config.GCP_BQ_LOCATION,
                                       dataset_name = Config.GCP_BQ_DATASET,
                                       table_name = Config.GCP_BQ_EMBEDS_TABLE,
                                       embedding = embedding_model,
                                       credentials = self.credentials)
        
        _ = bq_store.add_texts(texts = chunks,
                               metadatas = metadata)

    def process_docs(self,
                     doc_id:list[str],
                     cont_user_id:str) -> dict:
        """Executes the whole content processing to update the vector space with every given document.
        Inputs:
            - doc_id: the list with the ICD document IDs
            - cont_user_id: the ID of the ICD contract (for PV Assessment) OR the user (for document comparison)
        Returns:
            - report: the dictionary mentioning which documents were uploaded correctly or not"""
        
        # self.create_BQ_table(".".join([Config.GCP_DEV_PROJECT_ID, Config.GCP_BQ_DATASET, Config.GCP_BQ_EMBEDS_TABLE]))

        pdf_parser = AzureDocIntParser()
        success = []
        failed = []

        for i, doc in enumerate(self.docs):
            # print(f"Checking doc {i} out of {len(self.docs)-1}", end="\r")

            filename, file_extension = self.paths[i].split("/")[-1].rsplit(".")
            print("Got filename")

            try:
                pva_json_data = pdf_parser.get_azureDocAI_data(filename, file_extension, doc)
                print(pva_json_data)
                print("Got document data")

                chunks, metadata = self.create_text_chunks(pva_json_data, doc_id[i], cont_user_id)
                print("Got chunks")

                self.store_vector_data_BQ(chunks, metadata)
                print("Saved into BigQuery")

                success.append(filename)

            except Exception as e:
                print(e)
                try:
                    query = f"DELETE FROM `{Config.GCP_DEV_PROJECT_ID}.{Config.GCP_BQ_DATASET}.{Config.GCP_BQ_EMBEDS_TABLE}` WHERE doc_id = {doc_id[i]};"
                    _ = self.run_BQ_query(query)
                except Exception as f:
                    print(f)
                
                failed.append(filename)
        
        return {"status": len(self.docs)==len(success),
                "success": success,
                "failed": failed}
    
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
    
def get_raw_data(path:str):
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
    
# if __name__ == "__main__":
    
#     doc_id = ["sample_word_1", "sample_word_2"]
#     cont_user_id = "test_excel"

#     document_paths = ["./doc_1.docx",
#                       "./doc_2.docx"]
    
#     vs = VectorSpace(paths = document_paths,
#                      docs = [get_raw_data(doc) for doc in document_paths])
#     print("Vector space object created succesfully")
#     upload = vs.process_docs(doc_id = doc_id,
#                              cont_user_id = cont_user_id)
#     print(upload)