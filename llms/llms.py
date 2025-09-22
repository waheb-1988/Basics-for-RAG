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

# Hugging Face (local pipeline -> LangChain)
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

# API Keys (update names if yours differ)
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
    # LangChain message-like
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    # Fallback to string
    return str(response)

# Minimal adapter so OpenRouter model can be used in the same loop with .invoke()
class OpenRouterTongyiAdapter:
    def __init__(self, api_key: str, model_name: str = "alibaba/tongyi-deepresearch-30b-a3b"):
        if OpenAI is None:
            raise RuntimeError("openai library not available. pip install openai")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing.")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model = model_name

    def invoke(self, prompt: str):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

# -----------------------
# Tongyi (local) via transformers -> HuggingFacePipeline
# -----------------------
def build_tongyi_local_pipeline():
    """
    Load Alibaba-NLP/Tongyi-DeepResearch-30B-A3B locally and wrap in LangChain.
    Requires transformers>=4.45, accelerate>=0.34, and enough VRAM (or a quantized variant).
    """
    # version guard for qwen3_moe architecture
    import transformers as _tf
    from packaging import version as _V
    if _V.parse(_tf.__version__) < _V.parse("4.45.0"):
        raise RuntimeError(
            f"transformers=={_tf.__version__} too old for qwen3_moe. "
            "Run: pip install -U 'transformers>=4.45' 'accelerate>=0.34' einops sentencepiece safetensors"
        )

    model_name = "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
    trust_rc = os.getenv("TRUST_REMOTE_CODE", "1") not in ("0", "false", "False")

    print(f"[Tongyi] Loading {model_name} (this can take a while)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_rc)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",        # auto-place on available GPU(s)
        torch_dtype="auto",       # use bf16/fp16 if available
        trust_remote_code=trust_rc
    )

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.2,
    )

    # Wrap for LangChain so you can call .invoke()
    return HuggingFacePipeline(pipeline=gen_pipe)

# Try local Tongyi; if it fails (e.g., not enough VRAM or old transformers), we’ll try OpenRouter
tongyi_llm = None
try:
    tongyi_llm = build_tongyi_local_pipeline()
    print("[Tongyi] ✅ Local pipeline ready")
except Exception as e:
    print(f"[Tongyi] ⚠️ Local load failed: {e}")
    # Hosted fallback
    try:
        if openrouter_api_key:
            tongyi_llm = OpenRouterTongyiAdapter(openrouter_api_key)
            print("[Tongyi] ✅ Using OpenRouter hosted model")
        else:
            print("[Tongyi] ⚠️ No OPENROUTER_API_KEY; Tongyi will be skipped.")
    except Exception as ee:
        print(f"[Tongyi] ⚠️ OpenRouter fallback failed: {ee}")

# -----------------------
# Define chat models to test (ONLY real chat LLMs here)
# -----------------------
llms = {}

# Tongyi (local or OpenRouter) first if available
if tongyi_llm is not None:
    llms["Tongyi-DeepResearch-30B-A3B"] = tongyi_llm

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
    raise RuntimeError("No LLMs initialized. Check your API keys or Tongyi setup (local/OpenRouter).")

# -----------------------
# Questions to ask
# -----------------------
questions = [
    "generate name statistics KPI in a very short sentence",
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
            print(f"✅ {name} Response: {content[:160]}...")
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
