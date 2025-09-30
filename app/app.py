# app.py
import os
import streamlit as st
from pymilvus import connections, Collection
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------------
# Milvus connection
# ------------------------------
connections.connect(alias="default", host="127.0.0.1", port="19530")

COLLECTION_NAME = "my_docs"

# Create Milvus handle
collection = Collection(COLLECTION_NAME)

# ------------------------------
# Initialize OpenAI
# ------------------------------
chat = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# ------------------------------
# Helper: Retrieve context
# ------------------------------
def search_milvus(query: str, top_k: int = 3):
    """Search Milvus with query embeddings"""
    query_emb = embeddings.embed_query(query)

    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

    results = collection.search(
        data=[query_emb],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["text"],
    )
    docs = [hit.entity.get("text") for hit in results[0]]
    return docs


# ------------------------------
# Helper: Build prompt
# ------------------------------
def build_prompt(question: str, context_docs: list[str]):
    template = """You are an assistant. Use the context below to answer.

Context:
{context}

Question: {question}
Answer:"""
    context = "\n\n".join(context_docs) if context_docs else "No context found."
    prompt = ChatPromptTemplate.from_template(template)
    return prompt.format_messages(context=context, question=question)


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Milvus + OpenAI RAG", layout="wide")
st.title("🔎 Milvus + OpenAI Chatbot")

user_query = st.text_input("Ask me something:")

if st.button("Submit") and user_query:
    with st.spinner("Searching knowledge base..."):
        docs = search_milvus(user_query, top_k=3)

    with st.spinner("Generating answer..."):
        messages = build_prompt(user_query, docs)
        response = chat(messages)

    st.subheader("Answer:")
    st.write(response.content)

    if docs:
        st.subheader("🔎 Context Used")
        for d in docs:
            st.markdown(f"- {d}")
