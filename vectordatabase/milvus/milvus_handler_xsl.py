import numpy as np
from pathlib import Path
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from dotenv import load_dotenv
import os

# --------------- Configuration ---------------
# Load Environment Variables
load_dotenv()

# Milvus Configuration from .env
USE_LOCAL = os.getenv("USE_LOCAL", "False").lower() == "true"
LOCAL_MILVUS_URI = os.getenv("LOCAL_MILVUS_URI")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
COLLECTION_NAME = os.getenv("COLLECTION_NAME_1")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION"))

# --------------- Milvus Connection ---------------
def connect_to_milvus():
    """Connects to local or remote Milvus."""
    uri = LOCAL_MILVUS_URI if USE_LOCAL else MILVUS_URI
    user, password = (None, None) if USE_LOCAL else (MILVUS_USER, MILVUS_PASSWORD)

    try:
        connections.connect(alias="default", uri=uri, user=user, password=password)
        print(f"✅ Connected to Milvus ({'Local' if USE_LOCAL else 'Cloud'})!")
    except Exception as e:
        print(f"❌ Milvus connection failed: {e}")

# --------------- Create or Load Collection ---------------
def create_or_load_collection():
    """Creates a new Milvus collection if it doesn't exist."""
    if utility.has_collection(COLLECTION_NAME):
        print(f"✅ Collection '{COLLECTION_NAME}' already exists.")
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, description="Document embeddings")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Create an index for fast retrieval
    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
    collection.create_index("vector", index_params)
    collection.load()

    print(f"✅ Collection '{COLLECTION_NAME}' created and loaded.")
    return collection

# --------------- Excel Processing ---------------
def load_excel_with_docling(file_path):
    """Loads Excel files using Docling DocumentConverter."""
    try:
        converter = DocumentConverter()
        result = converter.convert(Path(file_path))
        output = result.document.export_to_markdown()

        # Use DoclingLoader and HybridChunker for better chunking
        loader = DoclingLoader(
            file_path=file_path,
            export_type=ExportType.DOC_CHUNKS,  # Corrected Export Type
            chunker=HybridChunker(tokenizer="bert-base-uncased"),
        )
        docs = loader.load()

        splits = docs
        print(f"✅ Loaded and split Excel file: {file_path}")
        return splits
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return None

# --------------- Embeddings & Storage ---------------

def embed_and_store_docs(docs, collection):
    """Embeds documents and stores them in Milvus."""
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings

    # Use HuggingFace embeddings (adjust model as needed)
    embedding_model = HuggingFaceEmbeddings(model_name="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja")
    
    vectors, texts = [], []

    for doc in docs:
        try:
            # Embed each chunk separately
            vector = embedding_model.embed_query(doc.page_content)
            vector = np.array(vector, dtype=np.float32)
            
            # Check if vector length matches expected dimension
            if vector.shape[0] != VECTOR_DIMENSION:
                print(f"❌ Vector dimension mismatch: Expected {VECTOR_DIMENSION}, but got {vector.shape[0]}")
                continue  # Skip this vector if dimension doesn't match
            
            vectors.append(vector.tolist())
            texts.append(doc.page_content)
        except Exception as e:
            print(f"❌ Error embedding document: {e}")

    if vectors:
        # Create Milvus insert data format
        insert_data = [
            {"vector": vectors[i], "text": texts[i]}
            for i in range(len(vectors))
        ]
        collection.insert(insert_data)
        print(f"✅ Inserted {len(vectors)} documents into Milvus.")
    else:
        print("❌ No valid vectors to insert.")


# --------------- Main Workflow ---------------
def main():
    file_path = r"C:\Abdelouaheb\perso\Data_science_2024_projects\2025\MacroEconmy_ChatBot\data\raw\test\Algeria_str.xlsx"

    # Connect to Milvus (local or cloud)
    connect_to_milvus()

    # Create or load the collection
    collection = create_or_load_collection()

    # Load and process Excel file
    docs = load_excel_with_docling(file_path)
    if docs:
        embed_and_store_docs(docs, collection)

if __name__ == "__main__":
    main()
