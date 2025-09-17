# run_query.py
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from pymilvus import connections
from milvus_handler_pdf import connect_to_milvus
# Load environment variables
load_dotenv()

# Configuration
USE_LOCAL = os.getenv("USE_LOCAL", "False").lower() == "true"
MILVUS_URI = os.getenv("LOCAL_MILVUS_URI") if USE_LOCAL else os.getenv("MILVUS_URI")
MILVUS_USER = None if USE_LOCAL else os.getenv("MILVUS_USER")
MILVUS_PASSWORD = None if USE_LOCAL else os.getenv("MILVUS_PASSWORD")
COLLECTION_NAME = os.getenv("COLLECTION_NAME_2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


def search_query(query, top_k=5):
    # Connect to Milvus and get connection arguments
    connection_args = connect_to_milvus()
    if not connection_args:
        print("⚠️ Aborting query due to connection error.")
        return

    # Create embedding model
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL)

    # Initialize vector store
    vectorstore = Milvus(
        embedding_function=embedding,
        connection_args=connection_args,
        collection_name=COLLECTION_NAME
    )

    # Perform similarity search
    results = vectorstore.similarity_search(query, k=top_k)

    print(f"\n🔍 Top {top_k} matches for your query:\n")
    for i, doc in enumerate(results):
        print(f"--- Result #{i+1} ---")
        print(f"Content:\n{doc.page_content}\n")
        print(f"Metadata: {doc.metadata}\n")
        print("-" * 40)


if __name__ == "__main__":
    user_query = input("📝 Enter your query: ")
    search_query(user_query, top_k=5)
