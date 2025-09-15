import os
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

# Load environment variables
load_dotenv()


class TabularProcessor:
    """
    A utility class to load, convert, and process tabular files
    (Excel .xls/.xlsx and CSV) into chunks for use in RAG pipelines.
    """

    def __init__(self, file_path, chunk_size=500, chunk_overlap=50):
        """
        Initialize the TabularProcessor.

        Args:
            file_path (str): Path to the Excel or CSV file.
            chunk_size (int): Maximum number of characters per chunk.
            chunk_overlap (int): Number of overlapping characters between chunks.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_path = self._ensure_supported_format(file_path)
        self.documents = None
        self.chunks = None

    def _ensure_supported_format(self, file_path):
        """
        Ensure the file is in a supported format.
        - Converts .xls → .xlsx for compatibility.
        - Converts .csv → .xlsx for compatibility.

        Args:
            file_path (str): Path to the file.

        Returns:
            str: Path to the converted file in .xlsx format.
        """
        try:
            if file_path.endswith(".xls"):
                output_path = file_path.replace(".xls", ".xlsx")
                df = pd.read_excel(file_path)
                df.to_excel(output_path, index=False)
                print(f"[INFO] Converted {file_path} → {output_path}")
                return output_path

            if file_path.endswith(".csv"):
                output_path = file_path.replace(".csv", ".xlsx")
                df = pd.read_csv(file_path)
                df.to_excel(output_path, index=False)
                print(f"[INFO] Converted {file_path} → {output_path}")
                return output_path

            return file_path
        except Exception as e:
            print(f"❌ Error converting file format: {e}")
            return file_path

    def load_file(self):
        """
        Load an Excel or CSV file using Docling.

        Returns:
            list: A list of document chunks.
        """
        try:
            loader = DoclingLoader(file_path=self.processed_path, export_type=ExportType.DOC_CHUNKS)
            self.documents = loader.load()
            print(f"[INFO] Loaded {len(self.documents)} raw chunks from {os.path.basename(self.processed_path)}")
            return self.documents
        except Exception as e:
            print(f"❌ Error loading file {self.processed_path}: {e}")
            return None

    def split_text(self):
        """
        Split loaded documents into smaller chunks.

        Returns:
            list: A list of split documents.
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call `load_file()` first.")

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            self.chunks = text_splitter.split_documents(self.documents)
            print(f"[INFO] Split documents into {len(self.chunks)} smaller chunks")
            return self.chunks
        except Exception as e:
            print(f"❌ Error splitting documents: {e}")
            return None

    def process(self):
        """
        Process a single file:
        - Convert if necessary
        - Load the file
        - Split it into chunks

        Returns:
            list: A list of chunked documents.
        """
        self.load_file()
        return self.split_text()

    @staticmethod
    def process_all(folder_path, chunk_size=500, chunk_overlap=50):
        """
        Process all Excel/CSV files in a given folder.

        Args:
            folder_path (str): Path to the folder containing files.
            chunk_size (int): Maximum number of characters per chunk.
            chunk_overlap (int): Number of overlapping characters between chunks.

        Returns:
            dict: A dictionary mapping filenames to their chunked documents.
        """
        results = {}
        supported_files = [
            f for f in os.listdir(folder_path)
            if f.endswith((".xls", ".xlsx", ".csv"))
        ]

        if not supported_files:
            print("[INFO] No Excel/CSV files found in the folder.")
            return results

        for file in supported_files:
            file_path = os.path.join(folder_path, file)
            processor = TabularProcessor(file_path, chunk_size, chunk_overlap)
            chunks = processor.process()
            if chunks:
                results[file] = {
                    "chunks": chunks,
                    "num_chunks": len(chunks),
                    "path": file_path
                }
                print(f"[INFO] Processed {file}: {len(chunks)} chunks created")

        return results


# ---------------- EXAMPLES ----------------
#  if __name__ == "__main__":
    # Example 1: Process a single file (Excel or CSV)
    # processor = TabularProcessor("path/to/your/file.csv")
    # chunks = processor.process()

    # Example 2: Process all Excel/CSV files in a folder
    # all_chunks = TabularProcessor.process_all("path/to/your/folder")
    # print(all_chunks.keys())
