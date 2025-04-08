import os
from secret_manager import secret_manager_key_retrieve

class Config:
    GCP_CREDENTIALS = secret_manager_key_retrieve("project-id", "gcp-sa-secret-manager-id")
    GCP_DEV_PROJECT_ID = "project-id"
    
    GCP_BQ_LOCATION = "us"
    GCP_BQ_DATASET = "dataset-name"
    GCP_BQ_EMBEDS_TABLE = "bq-table-name"

    GCP_CS_BUCKET = "gcp-bq-bucket"

    GEMINI_EMBEDDING_MODEL = "text-embedding-004"
    GEMINI_LLM_VERSION = "gemini-1.5-pro-002"
    GEMINI_MAX_OUTPUT_TOKENS = 8192
    GEMINI_TEMPERATURE = 0.2
    GEMINI_TOP_P = 0.95

    GCP_LLM_SQL_PROMPT = """You are a Pharmacovigilance contract professional who needs to compare two different excel sheets with data from multiple contracts.
    Based on the given context, generate only a list with the contracts that match the user request.
    Do NOT generate any summary or explanation.
    
    """

    LANGCHAIN_RETRIEVER_SEARCH_TYPE = "mmr" # Similarity or mmr
    LANGCHAIN_RETRIEVER_NUMBER_RESULTS = 10
    LANGCHAIN_DOC_COMPARISON_BASE_PROMPT = """You are a Pharmacovigilance contract professional who needs to compare a partner document with a given template.
You will be given a user request that needs to be answered based on the content of the given documents.
Your answer MUST be always a single, brief paragraph.
You will ALWAYS have two uploaded documents.
You will ALWAYS have one business template and one real contract.
Consider which document is the business template and which one the real contract.
Do NOT generate any tables on your response.
Only use text for your answer.
As long as the content is present on the document, it is ok if it's present on a different section than the one mentioned on the user query.
If the wording is slightly different but the meaning of it/responsibility described is the same this would mean the clause is aligned.
If the information is not present, like an explicit address or number, just mention that you were 'not able to find that information based on the information retrieved'.

-- Documents classification --
{classification}

-- User query --
{user_query}

-- Documents content --
{docs_content}"""

    LANGCHAIN_DOC_COMPARISON_CHAT_HISTORY_PROMPT = """Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is."""

    SQL_CSV_PROMPTS = ["Evaluation of Common Contracts",
                       "Evaluation of Terminated Contracts",
                       "Evaluation of Newly Added Contracts"]
    SQL_COMMON_CONTRACTS = """WITH Doc1 AS(
        SELECT DISTINCT doc_id, filename, section, page_row, content
        FROM `{}.{}.{}`
        WHERE contract_user_id = '{}'
            AND doc_id = '{}'),
    Doc2 AS(
        SELECT DISTINCT doc_id, filename, section, page_row, content
        FROM `{}.{}.{}`
        WHERE contract_user_id = '{}'
            AND doc_id = '{}')
SELECT DISTINCT Doc1.doc_id, Doc1.filename, Doc1.section, Doc1.page_row, Doc1.content
FROM Doc1
JOIN Doc2 ON Doc1.section = Doc2.section"""
    SQL_DIFFERENT_CONTRACTS = """WITH Doc1 AS(
            SELECT DISTINCT doc_id, filename, section, page_row, content
            FROM `{}.{}.{}`
            WHERE contract_user_id = '{}'
                AND doc_id = '{}'),
        Doc2 AS(
            SELECT DISTINCT doc_id, filename, section, page_row, content
            FROM `{}.{}.{}`
            WHERE contract_user_id = '{}'
                AND doc_id = '{}')
    SELECT DISTINCT Doc1.doc_id, Doc1.filename, Doc1.section, Doc1.page_row, Doc1.content
    FROM Doc1
    LEFT JOIN Doc2 ON Doc1.section = Doc2.section
    WHERE Doc2.section IS NULL
    ORDER BY Doc1.section ASC"""

    LANGCHAIN_SQL_SIMILARITY_QUERY = """SELECT doc_id, filename, section, page_row, content, ML.DISTANCE(embedding, {query_emb}) as distance
FROM `bq-dataset-name`
WHERE doc_id = '{doc_1}' OR doc_id = '{doc_2}'
ORDER BY distance DESC
LIMIT {k}"""

    LANGCHAIN_SQL_GET_DOCS_DATA = """SELECT doc_id, filename, section, page_row, content
FROM `bq-dataset-name`
WHERE doc_id = '{doc_1}' OR doc_id = '{doc_2}'"""

    LANGCHAIN_CLASSIFY_DOCUMENTS_MSG = """You are a Pharmacovigilance contract professional who needs to compare a partner document with a given template.
Based on the content of the given documents, determine which document is the base template and which one the partner document.
There will ALWAYS be ONE template and ONE document.
The same document CANNOT be both the template and the document.
If the references talk more about general requirements, then that document is the template.
If the references talk about agreements between two parties or provide concrete, non-generic information like product names, amounts of money, partnerships between specific companies, etc., then that document is the partner document.
DO NOT explain your decision.
ONLY tell which document is the template and which the contract.

Uploaded documents:
{uploaded_docs}"""

    AZR_DOC_INT_ENDPOINT_SECRET_ID = "azr-doc-int-endpoint-secret-manager-id"
    AZR_DOC_INT_LOCATION_SECRET_ID = "azr-doc-int-location-secret-manager-id"
    AZR_DOC_INT_KEY_SECRET_ID = "azr-doc-int-key-secret-manager-id"