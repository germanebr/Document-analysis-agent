import os
import requests
import json
import vertexai.preview.generative_models as generative_models
from typing import List, Optional
from google.cloud import aiplatform, bigquery
from google.oauth2 import service_account
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

from config import Config

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GCP_CREDENTIALS

def init_gcp():
    credentials = service_account.Credentials.from_service_account_file(Config.GCP_CREDENTIALS)

    aiplatform.init(project = "gcp-project-id",
                    credentials = credentials)
    
    return credentials
    
def init_bq():
    credentials = init_gcp()
    
    client = bigquery.Client(project = 'gcp-project-id',
                             credentials = credentials)
    return client
    
def create_BQ_table(table_id:str = Config.GCP_BQ_EMBEDS_TABLE):
    """Creates BigQuery table with table_id provided, or if table already exists, does nothing.
    Inputs:
        - table_id: the GCP uri for creating the table
    Returns:
        - none"""

    client = init_bq()

    schema = [bigquery.SchemaField("doc_id", "STRING", mode = "REQUIRED"),
              bigquery.SchemaField("section", "STRING", mode = "REQUIRED"),
              bigquery.SchemaField("page", "INTEGER", mode = "REQUIRED"),
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

def run_BQ_query(query:str) -> List:
    """Runs the given SQL query through the BigQuery microservice.
    Please modify to the corresponding code according to documentation or the team's needs.
    Inputs:
        - query: the BigQuery query to execute
    Returns:
        - res: the response obtained"""

    url = "microservice url for connecting to GCP BigQuery"
    response = requests.post(url,
                             json = {"query": query,
                                     "records_per_page": 1000})
    # print(response.json())
    return json.loads(response.json()["res"])["data"]

def upload_file_BQ(table_id:str,
                   headers:List[str],
                   values:List[str]):
    """Upload the document data to BigQuery
    Inputs:
        - table_id: the BigQuery table uri where the data will be stored
        - headers: the column names of the BigQuery table
        - values: the list of values to insert on every row
    Returns:
        - none"""
    
    client = init_bq()

    heads = ", ".join(headers)
    vals = ", ".join(values)

    query = f"""INSERT INTO `{table_id}` ({heads}) VALUES {vals};"""

    try:
        client.query_and_wait(query)
    except Exception as e:
        with open("query_fail.txt", "a", encoding = "utf-8") as f:
            f.write(query + "\n\n")
        raise Exception(e)
    
def generate_gcp_embeddings(texts:List[str],
                            task:str = "RETRIEVAL_DOCUMENT",
                            dimensionality:Optional[int] = 768) -> List[List[float]]:
    """Generate the embedding of the given text.
    Inputs:
        - texts: the string to convert
        - task: the type of embedding that needs to be generated (https://ai.google.dev/gemini-api/tutorials/anomaly_detection)
        - dimensionality: the number of dimensions to generate in the embedding
    Returns:
        - embedding: the generated text embedding"""
    
    model = TextEmbeddingModel.from_pretrained(Config.GEMINI_EMBEDDING_MODEL)
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    kwargs = dict(output_dimensionality = dimensionality) if dimensionality else {}
    embeddings = []
    chunk_size = len(texts)
    completed = False

    while (not completed) and (chunk_size >= 1):
        # print(f"\t> Splitting into {math.ceil(len(inputs)/chunk_size)} chunks.")
        chunks = [inputs[i:i + chunk_size] for i in range(0, len(inputs), chunk_size)]

        try:
            for i,chunk in enumerate(chunks):
                # print(f"\t> Embeddings {(100*i)/len(chunks):0.2f}% completed", end="\r")
                embeddings += model.get_embeddings(chunk, **kwargs)

            completed = True
        
        except Exception as e:
            # print(f"\n\t> Chunks were too big. Retrying with more chunks.\n", end="\r")
            chunk_size = int(chunk_size // 2)

            if chunk_size < 1:
                raise Exception(e)
        
    # print(f"\t> Embeddings 100.00% completed")
    return [embedding.values for embedding in embeddings]

def initialize_gemini(model_name:str = Config.GEMINI_LLM_VERSION,
                      agent_prompt:str = "",
                      temp:int = 0,
                      max_tkn:int = 8192,
                      p:float = 0.95) -> GenerativeModel:
    """Initialize the LLM to get resopnses from it.
    Inputs:
        - model_name: the name of the LLM to use
        - agent_prompt: the base prompt for the LLM to follow
        - temp: the LLM temperature
        - max_tkn: maximum number of output tokens of the LLM
        - p: answer selection probability for the LLM
    Returns:
        - model: the initizalized LLM"""
    
    init_gcp()

    safety_settings = {generative_models.HarmCategory.HARM_CATEGORY_UNSPECIFIED: generative_models.HarmBlockThreshold.BLOCK_NONE,
                       generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                       generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
                       generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                       generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE}
    
    model = GenerativeModel(model_name,
                            generation_config = {"temperature": temp,
                                                 "max_output_tokens": max_tkn,
                                                 "top_p": p},
                            system_instruction = agent_prompt,
                            safety_settings = safety_settings)
    return model

def count_gcp_tokens(text:str) -> int:
    """Count how many tokens will be given to the Gemini LLM
    Inputs:
        - text: the string to evaluate
    Returns:
        - tokens: number of tokens of the given text"""
    
    model = initialize_gemini()

    return model.count_tokens([text]).total_tokens