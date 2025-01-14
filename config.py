import os

class Config:
    GCP_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'service_account.json')
    GCP_DEV_PROJECT_ID = "gcp-project"
    
    GCP_BQ_LOCATION = "us"
    GCP_BQ_DATASET = "bq_dataset"
    GCP_BQ_EMBEDS_TABLE = "bq_table"

    GCP_CS_BUCKET = "gcp-cs-bucket"

    GEMINI_EMBEDDING_MODEL = "text-embedding-004"
    GEMINI_LLM_VERSION = "gemini-1.5-pro-002"
    GEMINI_MAX_OUTPUT_TOKENS = 8192
    GEMINI_TEMPERATURE = 0.2
    GEMINI_TOP_P = 0.95

    LANGCHAIN_RETRIEVER_SEARCH_TYPE = "mmr" # Similarity or mmr
    LANGCHAIN_RETRIEVER_NUMBER_RESULTS = 5
    LANGCHAIN_DOC_COMPARISON_BASE_PROMPT = """You are a Pharmacovigilance contract professional who needs to compare a partner document with a template.
You will be given a set of key features with their corresponding references to answer the given user request.
Use only the provided information to generate your answer.
Consider which document is the business template and which one the real contract.
Do NOT generate any tables on your response.
Only use text for your answer.
Always end your response with a final conclusion that summarizes everything you said.

-- Documents --
{classification}

-- User query --
{user_query}

-- Key Features --
{key_features}

-- References --
{references}"""
    LANGCHAIN_DOC_COMPARISON_CHAT_HISTORY_PROMPT = """Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is."""

    SQL_VST_PROMPTS = ["Evaluation of Common Contracts",
                       "Evaluation of Terminated Contracts",
                       "Evaluation of Newly Added Contracts"]
    SQL_PSMF_COMMON_CONTRACTS = """WITH Doc1 AS(
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
    SQL_PSMF_DIFFERENT_CONTRACTS = """WITH Doc1 AS(
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

    AZURE_DOC_INTELLIGENCE_KEY = "{azure_document_intelligence_api_key}"

    LANGCHAIN_KEYWORD_EXTRACTOR_MSG = """You will be given a user request to analyze two different documents.
Your goal is to generate a well-structured query for use in retrieval related to that request.
First, analyze the request.
Pay particular attention to the key elements that need to be retrieved ONLY from the business contract.
Convert these key elements into a well-structured phrase indicating what needs to be retrieved.
Generate maximum four key elements.
ONLY give me the list of key elements.
                                      
User request:
{query}"""

    LANGCHAIN_SQL_SIMILARITY_QUERY = """SELECT doc_id, filename, section, page_row, content, ML.DISTANCE(embedding, {query_emb}) as distance
FROM `gcp-vpcx-acl.pva_analysis.docs_vector_store_test`
WHERE doc_id = '{doc_1}' OR doc_id = '{doc_2}'
ORDER BY distance DESC
LIMIT {k}"""

    LANGCHAIN_CLASSIFY_DOCUMENTS_MSG = """You are a Pharmacovigilance contract professional who needs to compare a partner document with a template.
Based on the given references, determine which document is the base template and which one the partner document.
There will ALWAYS be ONE template and ONE document.
The same document CANNOT be both the template and the document.
If the references talk more about general requirements, then that document is the template.
If the references talk about agreements between two parties or provide concrete, non-generic information like product names, amounts of money, partnerships between specific companies, etc., then that document is the partner document.
DO NOT explain your decision.
ONLY tell which document ID is the template and which the contract.
The document IDs are the key values of the given references.

References:
{references}"""