# Optionally save to CSV/Excel
import os
import time
import pandas as pd
from dotenv import load_dotenv

# -----------------------
# Import LLMs
# -----------------------
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms import HuggingFaceHub
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# -----------------------
# Load environment variables
# -----------------------
load_dotenv()

# API Keys
mistral_api_key = os.getenv("MISTRAL_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY1")
deepseek_api_key = os.getenv("DEEP_API_KEY")
hug_api_key = os.getenv("HUG_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")
groq_llm_model = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")

# -----------------------
# Define models to test
# -----------------------
llms = {
    "MistralAI": ChatMistralAI(mistral_api_key=mistral_api_key),
    "OpenAI (GPT-4o-mini)": ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-4o-mini",
        temperature=0.0
    ),
    "DeepSeek": ChatDeepSeek(
        model="deepseek-chat",
        api_key=deepseek_api_key
    ),
    # Replace HuggingFace Zephyr with bge reranker
    "HuggingFace (BGE-Reranker)": HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-reranker-v2-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    ),
    "Groq": ChatGroq(api_key=groq_api_key, model=groq_llm_model)
}

# -----------------------
# Questions to ask
# -----------------------
questions = [
    "What are the key benefits of AI in healthcare?",
    "What is LLM?"
]

# -----------------------
# Run tests
# -----------------------
results = []
for q in questions:
    for name, llm in llms.items():
        print(f"\n🛠️ Testing {name} on question: {q}")
        try:
            start_time = time.time()
            response = llm.invoke(q)
            elapsed = time.time() - start_time

            # HuggingFace embeddings don't return text → skip answering
            if isinstance(response, str):
                content = response
            elif hasattr(response, "content"):
                content = response.content
            else:
                content = "[Embedding Model - no text answer]"

            print(f"✅ {name} Response: {content[:120]}...")
            results.append([q, name, content, elapsed])
        except Exception as e:
            print(f"❌ {name} failed: {str(e)}")
            results.append([q, name, f"Error: {str(e)}", None])

# -----------------------
# Show comparison table
# -----------------------
df = pd.DataFrame(results, columns=["Question", "Model", "Answer", "Response Time (s)"])
print("\n🔍 **LLM Benchmark Results:**")
print(df.to_string(index=False))


