import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from typing import List
import os

class ExcelLoader(BaseLoader):
    """
    Loads data from an Excel file, handling multiple sheets.
    Each row (question, answer) in each sheet becomes a LangChain Document.
    The content of a document is "Question: {question}\nAnswer: {answer}".
    The document's metadata includes the base filename as a 'tag' key.
    """
    def __init__(self, file_path: str):
        """
        Initializes the ExcelLoader with the path to the Excel file.

        Args:
            file_path (str): The path to the Excel file.
        """
        self.file_path = file_path

    def load(self) -> List[Document]:
        """
        Loads data from the Excel file and converts it into a list of LangChain Document objects.

        Returns:
            List[Document]: A list of LangChain Document objects, where each document
                            represents a row from the Excel file.
        """
        all_documents = []
        try:
            # Read all sheets from the Excel file into a dictionary of DataFrames
            # sheet_name=None will return a dictionary where keys are sheet names
            # and values are the corresponding DataFrames.
            excel_data = pd.read_excel(self.file_path, sheet_name=None)
        except Exception as e:
            print(f"Error reading Excel file {self.file_path}: {e}")
            return []

        for sheet_name, df in excel_data.items():
            # Ensure 'question' and 'answer' columns exist
            if 'question' not in df.columns or 'answer' not in df.columns:
                print(f"Warning: Sheet '{sheet_name}' in '{self.file_path}' "
                    f"does not contain 'question' or 'answer' columns. Skipping.")
                continue

            for index, row in df.iterrows():
                question = str(row['question']).strip() if pd.notna(row['question']) else ""
                answer = str(row['answer']).strip() if pd.notna(row['answer']) else ""

                # Construct the content for the document
                content = f"Question: {question}\nAnswer: {answer}"

                metadata = {
                    "sheet_name": sheet_name,
                    "row_index": index,
                    "source": self.file_path
                }

                doc = Document(page_content=content, metadata=metadata)
                all_documents.append(doc)

        print(f"Successfully loaded {len(all_documents)} documents from {self.file_path}")
        return all_documents


if __name__ == "__main__":
    # Create a dummy Excel file with multiple sheets for testing
    dummy_data_sheet1 = {
        'question': ["What is RAG?", "How does LangChain work?"],
        'answer': ["RAG combines retrieval with generation.", "LangChain provides tools for LLM apps."]
    }
    dummy_df_sheet1 = pd.DataFrame(dummy_data_sheet1)

    dummy_data_sheet2 = {
        'question': ["What is Ollama?", "What is ChromaDB?"],
        'answer': ["Ollama runs LLMs locally.", "ChromaDB is a local vector store."]
    }
    dummy_df_sheet2 = pd.DataFrame(dummy_data_sheet2)

    # Create 'data' directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    dummy_excel_path = "data/multi_sheet_data.xlsx"

    with pd.ExcelWriter(dummy_excel_path) as writer:
        dummy_df_sheet1.to_excel(writer, sheet_name='General Info', index=False)
        dummy_df_sheet2.to_excel(writer, sheet_name='Tech Details', index=False)

    print(f"Dummy Excel file created at: {dummy_excel_path}")

    # Test the ExcelLoader
    loader = ExcelLoader(file_path=dummy_excel_path)
    documents = loader.load()

    for doc in documents:
        print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}\n---")

    # Clean up the dummy file
    # os.remove(dummy_excel_path)
    # print(f"Dummy Excel file removed: {dummy_excel_path}")