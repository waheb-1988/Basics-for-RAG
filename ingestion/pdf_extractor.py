import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFProcessor:
    """
    A utility class to load, split, and process PDF files into chunks
    for use in Retrieval-Augmented Generation (RAG) pipelines.
    """

    def __init__(self, data_path, chunk_size=500, chunk_overlap=50):
        """
        Initialize the PDFProcessor.

        Args:
            data_path (str): Path to the PDF file or folder containing PDFs.
            chunk_size (int): Maximum number of characters per chunk.
            chunk_overlap (int): Number of overlapping characters between chunks.
        """
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_name = os.path.basename(data_path) if os.path.isfile(data_path) else None
        self.documents = None
        self.texts = None

    def load_pdf(self):
        """
        Load a single PDF file using PyPDFLoader.

        Returns:
            list: A list of documents (pages) extracted from the PDF.
        """
        loader = PyPDFLoader(self.data_path)
        self.documents = loader.load()
        print(f"[INFO] Loaded {len(self.documents)} pages from {self.pdf_name}")
        return self.documents

    def split_text(self):
        """
        Split the loaded PDF into smaller text chunks.

        Returns:
            list: A list of chunked documents.
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call `load_pdf()` first.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.texts = text_splitter.split_documents(self.documents)
        print(f"[INFO] Split documents into {len(self.texts)} chunks")
        return self.texts

    def process(self):
        """
        Process a single PDF file:
        - Load the PDF
        - Split it into chunks

        Returns:
            list: Chunked documents
        """
        self.load_pdf()
        return self.split_text()

    @staticmethod
    def process_all_pdfs(folder_path, chunk_size=500, chunk_overlap=50):
        """
        Process all PDF files in a given folder.

        Args:
            folder_path (str): Path to the folder containing PDF files.
            chunk_size (int): Maximum number of characters per chunk.
            chunk_overlap (int): Number of overlapping characters between chunks.

        Returns:
            dict: A dictionary mapping filenames to their chunked documents.
        """
        results = {}
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

        if not pdf_files:
            print("[INFO] No PDF files found in the folder.")
            return results

        for pdf in pdf_files:
            pdf_path = os.path.join(folder_path, pdf)
            processor = PDFProcessor(pdf_path, chunk_size, chunk_overlap)
            chunks = processor.process()
            results[pdf] = chunks
            print(f"[INFO] Processed {pdf}: {len(chunks)} chunks created")

        return results


# ---------------- EXAMPLES ----------------
# Example 1: Process a single PDF
# processor = PDFProcessor("path/to/your/file.pdf")
# chunks = processor.process()

# Example 2: Process all PDFs in a folder
# all_chunks = PDFProcessor.process_all_pdfs("path/to/your/pdf_folder")
# print(all_chunks.keys())
