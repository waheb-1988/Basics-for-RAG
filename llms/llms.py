# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
from dotenv import load_dotenv

# -----------------------
# LangChain providers
# -----------------------
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_groq import ChatGroq

# Hugging Face (local pipeline)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# -----------------------
# Optional hosted fallback (OpenRouter)
# -----------------------
try:
    from openai import OpenAI  # used only if OPENROUTER_API_KEY is set
except Exception:  # pragma: no cover
    OpenAI = None

# -----------------------
# Load environment variables
# -----------------------
load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY1")
deepseek_api_key = os.getenv("DEEP_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
groq_llm_model = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# -----------------------
# Helpers
# -----------------------
def normalize_content(response):
    """Return a plain string from LangChain/other client responses."""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    return str(response)

# -----------------------
# Hugging Face: very basic local model (distilgpt2)
# -----------------------
def build_hf_basic_pipeline():
    """
    Load a very small Hugging Face model (distilgpt2).
    Works on CPU, no API key required.
    """
    model_name = "distilgpt2"
    print(f"[HF] Loading {model_name} (tiny model, CPU friendly)...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=80,
        temperature=0.7,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)

# Try to load Hugging Face basic model
hf_llm = None
try:
    hf_llm = build_hf_basic_pipeline()
    print("[HF] ✅ Local Hugging Face (distilgpt2) ready")
except Exception as e:
    print(f"[HF] ⚠️ Failed to load Hugging Face model: {e}")

# -----------------------
# Define chat models
# -----------------------
llms = {}

# Hugging Face tiny model
if hf_llm is not None:
    llms["HuggingFace (distilgpt2)"] = hf_llm

# Mistral
if mistral_api_key:
    llms["MistralAI"] = ChatMistralAI(mistral_api_key=mistral_api_key)

# OpenAI
if openai_api_key:
    llms["OpenAI (gpt-4o-mini)"] = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4o-mini",
        temperature=0.0
    )

# DeepSeek
if deepseek_api_key:
    llms["DeepSeek (deepseek-chat)"] = ChatDeepSeek(
        model="deepseek-chat",
        api_key=deepseek_api_key
    )

# Groq
if groq_api_key:
    llms["Groq"] = ChatGroq(api_key=groq_api_key, model=groq_llm_model)

if not llms:
    raise RuntimeError("No LLMs initialized. Check API keys or Hugging Face setup.")

# -----------------------
# Questions to ask
# -----------------------
questions = [
    "give me statistic KPI",
   
]

# -----------------------
# Run tests
# -----------------------
results = []
for q in questions:
    for name, llm in llms.items():
        print(f"\n🛠️ Testing {name} on: {q}")
        try:
            start_time = time.time()
            response = llm.invoke(q)
            elapsed = time.time() - start_time
            content = normalize_content(response)
            print(f"✅ {name} Response: {content[:200]}...")
            results.append([q, name, content, round(elapsed, 3)])
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append([q, name, f"Error: {e}", None])

# -----------------------
# Show comparison table
# -----------------------
df = pd.DataFrame(results, columns=["Question", "Model", "Answer", "Response Time (s)"])
print("\n🔍 **LLM Benchmark Results:**")
print(df.to_string(index=False))
