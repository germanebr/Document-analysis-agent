import pandas as pd
import re
from azure.core.credentials import AzureKeyCredential #install azure-ai-documentintelligence
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from config import Config

class AzureDocIntParser():
    def __init__(self) -> None:
        endpoint = "{azure_document_intelligence_endpoint}"
            
        credential = AzureKeyCredential(Config.AZURE_DOC_INTELLIGENCE_KEY)
        self.document_intelligence_client = DocumentIntelligenceClient(endpoint, credential)

    def docAI_read_doc(self, data:bytes) -> AnalyzeResult:
        """Creates a readable object for Azure Document Intelligence from the document path ()
        Inputs:
            - data: the raw bytes of the document to read
        Returns:
            - doc: the Azure Doc Intelligence object with the file data"""
        
        poller = self.document_intelligence_client.begin_analyze_document("prebuilt-layout",
                                                                          analyze_request = data,
                                                                          content_type = "application/octet-stream")
         
        doc = poller.result()
        return doc
    
    def get_doc_text(self, doc:AnalyzeResult) -> dict:
        """Extracts the text from the given document
        Inputs:
            - doc: the Azure Doc Intelligence document
        Returns:
            - document_content: the dictionary with all the document text separated by section"""

        try:        
            document_content = {f"page_{i}": {} for i in range(1, doc.paragraphs[-1].bounding_regions[0]["pageNumber"] + 1)}
        except:
            document_content = {f"page_{i}": {} for i in range(1, len(doc.pages) + 1)}
        section = "Document Start"

        for paragraph in doc.paragraphs:
            # Get current page
            try:
                current_page = paragraph.bounding_regions[0]["pageNumber"]
            except:
                current_page = 1

            if paragraph.role not in ["pageFooter", "pageNumber"]:
                # Get latest role
                if paragraph.role:
                    # Use the new section
                    section = paragraph.content

                # Create the section dictionary on the page if it doesn't exist
                if section not in document_content[f"page_{current_page}"]:
                    # Store the paragraph
                    document_content[f"page_{current_page}"][section] = [paragraph.content]

                # Store the paragraph in the section if it already exists
                else:
                    document_content[f"page_{current_page}"][section].append(paragraph.content)
        
        # print("\t* Finished evaluating the document's text")
        return document_content
    
    def get_doc_tables(self, doc:AnalyzeResult) -> dict:
        """Extracts the data from the given document's tables
        Inputs:
            - doc: the Azure Doc Intelligence document
        Returns:
            - tables: the dictionary with all the table's data"""
        
        if doc.tables:
            tables = {f'table_{i}': '' for i in range(len(doc.tables))}

            for table_idx, table in enumerate(doc.tables):
                # print(f"\t* Analyzing table {table_idx}: {((100*table_idx)/len(doc.tables)):0.2f}% done", end = "\r")

                rows = [[] for _ in range(table.row_count)]

                for cell in table.cells:
                    rows[cell.row_index].append(cell.content.translate(str.maketrans('\"', ' ', '"\n\\')).replace("  ", " "))
                    
                rows = [" | ".join(row) for row in rows]

                try:
                    tables[f"table_{table_idx}"] = {"page": table.bounding_regions[0].page_number,
                                                    "content": " \\n".join(rows)}
                except:
                    tables[f"table_{table_idx}"] = {"page": 1,
                                                    "content": " \\n".join(rows)}
        
            # print("\t* Finished evaluating the document's tables")   
            return tables
        else:
            return {}
    
    def get_excel_text(self, doc:pd.DataFrame) -> dict:
        """Extracts the text from the given tabular document
        Inputs:
            - doc: the pandas dataframe with the document data
        Returns:
            - document_content: the dictionary with all the document text separated by section"""
        
        document_content = {f"row_{i}": {} for i in range(1, len(doc) + 1)}

        # Separate the content based on the column with the contract IDs
        # The code was developed for a particular Excel schema. Please modify as necessary
        icd_col = "Main header name"
        col_content = ["Header_1", "Header_2", "Header_3"]

        # Organize the data based on the selected columns
        for i, row in doc.iterrows():
            try:
                section = re.sub(r"\s\s+", " ", str(row[icd_col]))
            except:
                section = re.sub(r"\s\s+", " ", str(row.iloc[0]))
            content = [f"{sec}: {row[sec]}" for sec in doc.columns]
            document_content[f"row_{i+1}"] = {section: content}

        return document_content

    def get_azureDocAI_data(self,
                            filename:str,
                            extension:str,
                            data) -> dict:
        """Use of Azure Document Intelligence to extract data from a document and parse it into a dictionary
        Inputs:
            - filename: the filename of the document to analyze
            - extension: the file extension of the document to analyze
            - data: the raw bytes or the pandas dataframe of the document
        Returns:
            - doc_data: the dictionary with the document's data"""

        # Processing for excel files
        if extension in ["xlsx", "csv"]:
            doc_text = self.get_excel_text(data)
            doc_tables = {}

        # Processing for other document types
        else:
            # Create the Azure Doc Intelligence object
            # print("Analyzing document through Azure Document Intelligence...")
            document = self.docAI_read_doc(data)
            print("\t* Document analyzed")

            # Get the text data
            print("Extracting document text...")
            doc_text = self.get_doc_text(document)
            print("\t> Done!")

            # Get the table's data
            print("Extracting document tables...")
            doc_tables = self.get_doc_tables(document)
            print("\t> Done!")

        print(doc_text)
        print()
        print(doc_tables)

        return {"filename": filename,
                "doc_text": doc_text,
                "doc_tables": doc_tables}

# if __name__ == "__main__":
#     parser = AzureDocIntParser()

#     file = './sample_doc.pdf'

#     doc = parser.get_azureDocAI_data(file)
#     print(doc)

#     doc = parser.docAI_read_doc(file)
#     tables = parser.get_doc_tables(doc)
#     print(tables)