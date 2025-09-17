import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from pathlib import Path
# --------------- Configuration ---------------
# Load Environment Variables
load_dotenv()

# Milvus Configuration from .env
USE_LOCAL = os.getenv("USE_LOCAL", "False").lower() == "true"
LOCAL_MILVUS_URI = os.getenv("LOCAL_MILVUS_URI")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
COLLECTION_NAME = os.getenv("COLLECTION_NAME_2")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION"))

# OpenAI Configuration from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
# --------------- Milvus Connection ---------------
def connect_to_milvus():
    """Connects to local or remote Milvus."""
    try:
        if MILVUS_URI.startswith("tcp://"):
            uri_clean = MILVUS_URI.replace("tcp://", "")
            host, port = uri_clean.split(":")
            connections.connect(alias="default", host=host, port=port)
            print("✅ Connected to local Milvus!")
            return {
                "host": host,
                "port": port,
                "secure": False
            }
        else:
            connections.connect(alias="default", uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD, secure=True)
            print("✅ Connected to cloud Milvus!")
            return {
                "uri": MILVUS_URI,
                "user": MILVUS_USER,
                "password": MILVUS_PASSWORD,
                "secure": True
            }

    except Exception as e:
        print(f"❌ Milvus connection failed: {e}")
        return None


# --------------- Create or Load Collection ---------------
def create_or_load_collection():
    """Creates a new Milvus collection if it doesn't exist."""
    if utility.has_collection(COLLECTION_NAME):
        print(f"✅ Collection '{COLLECTION_NAME}' already exists.")
        return Collection(COLLECTION_NAME)

    fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=500),
]
    schema = CollectionSchema(fields, description="Document embeddings")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Create an index for fast retrieval
    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
    collection.create_index("vector", index_params)
    collection.load()

    print(f"✅ Collection '{COLLECTION_NAME}' created and loaded.")
    return collection


# --------------- PDF Processing ---------------
def load_files_with_metadata(path_folder):
    """Loads PDFs and extracts metadata."""
    try:
        loader = DirectoryLoader(path_folder, glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        metadata = [{"filename": os.path.splitext(os.path.basename(doc.metadata['source']))[0]} for doc in docs]
        return docs, metadata
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return None, None


def split_documents(docs, chunk_size=500, overlap=50):
    """Splits documents into smaller chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return text_splitter.split_documents(docs)
    except Exception as e:
        print(f"❌ Error splitting documents: {e}")
        return None


# --------------- Embeddings & Storage ---------------
def embed_and_store_docs(docs, metadata, collection):
    """Embeds documents and stores them in Milvus."""
    embedding_model = OpenAIEmbeddings(model=OPENAI_MODEL, openai_api_key=OPENAI_API_KEY)
    
    vectors, texts = [], []

    for doc in docs:
        try:
            vector = embedding_model.embed_query(doc.page_content)
            vector = np.array(vector, dtype=np.float32).tolist()  # Ensure correct format
            vectors.append(vector)
            texts.append(doc.page_content)
        except Exception as e:
            print(f"❌ Error embedding document: {e}")

    if vectors:
        insert_data = [{"vector": vectors[i], "text": texts[i], "filename": metadata[i]["filename"]} for i in range(len(vectors))]


        collection.insert(insert_data)
        print(f"✅ Inserted {len(vectors)} documents into Milvus.")


def create_local(collection_name, docs, metadata):
    """Creates a local Milvus collection, embeds documents, and inserts them."""
    connections.connect(alias="local", uri=LOCAL_MILVUS_URI)  # Use separate alias

    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        print(f"✅ Collection '{collection_name}' already exists, loading it.")
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
        ]
        schema = CollectionSchema(fields, description="Local vector DB")
        collection = Collection(name=collection_name, schema=schema)
        print(f"✅ Created new collection '{collection_name}'.")

    embed_and_store_docs(docs, metadata, collection)
    collection.flush()
    print(f"✅ Saved {len(docs)} vectors locally in '{collection_name}'")

def load_local(collection_name):
    """Loads a local Milvus collection."""
    connections.connect(alias="local", uri=LOCAL_MILVUS_URI)  # Use separate alias
    collection = Collection(name=collection_name)
    collection.load()
    print(f"✅ Loaded collection '{collection_name}'")
    return collection
def align_metadata_to_chunks(original_docs, metadata, chunk_size=500, overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs_split = []
    metadata_split = []
    
    for i, doc in enumerate(original_docs):
        chunks = text_splitter.split_documents([doc])
        docs_split.extend(chunks)
        metadata_split.extend([metadata[i]] * len(chunks))
        
    return docs_split, metadata_split

def delete_pdf_from_collection(collection_name, filename_keyword):
    """
    Deletes all vector entries related to a specific PDF by filename (without .pdf extension).
    Requires 'filename' field in Milvus schema.
    """
    try:
        connect_to_milvus()
        collection = Collection(collection_name)
        collection.load()

        expr = f'filename == "{filename_keyword}"'
        print(f"🗑️ Deleting entries where: {expr}")
        collection.delete(expr)
        collection.flush()

        print(f"✅ Deleted all chunks from PDF '{filename_keyword}' in collection '{collection_name}'")
    except Exception as e:
        print(f"❌ Error deleting from collection: {e}")



def add_pdf_to_existing_collection(file_path, collection):
    try:
        if not Path(file_path).is_file():
            raise ValueError(f"File path {file_path} is not a valid file.")

        # Load and split
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs_split = splitter.split_documents(docs)

        # Metadata
        file_name = Path(file_path).stem
        metadata = [{"source": file_name, "filename": file_name}] * len(docs_split)

        # Embed and store (pass collection, not name)
        embed_and_store_docs(docs_split, metadata, collection)
        print("✅ PDF successfully added to collection")

    except Exception as e:
        print(f"❌ Error adding PDF to collection: {e}")



def delete_collection(collection_name):
    """Deletes a collection from Milvus."""
    try:
        connect_to_milvus()
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"🗑️ Collection '{collection_name}' has been deleted.")
        else:
            print(f"⚠️ Collection '{collection_name}' does not exist.")
    except Exception as e:
        print(f"❌ Error deleting collection: {e}")

# --------------- Main Workflow ---------------

# def main():
#     path_folder = r"C:\Abdelouaheb\perso\Data_science_2024_projects\2025\MacroEconmy_ChatBot\data\raw\pdf"

#     # Connect to Milvus (local or cloud)
#     connect_to_milvus()

#     # Create or load the collection
#     collection = create_or_load_collection()

#     # Load and process documents
#     docs, metadata = load_files_with_metadata(path_folder)
#     if docs and metadata:
#         docs_split = []
#         metadata_split = []
#         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

#         for i, doc in enumerate(docs):
#             chunks = splitter.split_documents([doc])
#             docs_split.extend(chunks)
#             metadata_split.extend([metadata[i]] * len(chunks))

#         if docs_split:
#             embed_and_store_docs(docs_split, metadata_split, collection)

#             # Save locally if needed
#             if USE_LOCAL:
#                 create_local(COLLECTION_NAME, docs_split, metadata_split)
# if __name__ == "__main__":
#     main()

# --------------- Delete Workflow ---------------
def main():
# Connect to Milvus (local or cloud)
    connect_to_milvus()
   
    
    # Create or load the actual collection (not just its name)
    # collection = create_or_load_collection()
    
    #path_file = r"C:\Abdelouaheb\perso\Data_science_2024_projects\2025\MacroEconmy_ChatBot\data\raw\test\file_67bd0662c91f671dd87f0fd1.pdf"
    #add_pdf_to_existing_collection(path_file, collection)
    
    delete_pdf_from_collection(COLLECTION_NAME, "file_67bd0662c91f671dd87f0fd1")


if __name__ == "__main__":
    main()