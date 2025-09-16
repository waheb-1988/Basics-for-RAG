import time
import os
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_groq import ChatGroq  # ✅ Groq integration


class EmbeddingTester:
    def __init__(self, text="The economy is improving."):
        load_dotenv()
        self.text = text
        # OpenAI
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
        # Mistral
        self.MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.MISTRAIL_EMBEDING_MODEL = os.getenv("MISTRAL_EMBEDDING_MODEL")
        # Groq
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.GROQ_EMBEDING_MODEL = os.getenv("GROQ_EMBEDING_MODEL")

        self.results = []
        self.hf_models = [
            ("BAAI/bge-large-en", {'device': 'cpu'}, {'normalize_embeddings': False}),
            ("BAAI/bge-m3", {'device': 'cpu'}, {'normalize_embeddings': False}),
            ("BAAI/bge-reranker-v2-m3", {'device': 'cpu'}, {'normalize_embeddings': False}),
            ("sentence-transformers/all-MiniLM-L6-v2", {'device': 'cpu'}, {'normalize_embeddings': False}),
            ("intfloat/e5-large-v2", {'device': 'cpu'}, {'normalize_embeddings': False}),
            ("thenlper/gte-small", {'device': 'cpu'}, {'normalize_embeddings': False}),
            ("Alibaba-NLP/gte-large-en-v1.5", {'device': 'cpu'}, {'normalize_embeddings': False})
        ]

    def test_huggingface_models(self):
        for model_name, model_kwargs, encode_kwargs in self.hf_models:
            print(f"\n🔍 Testing HuggingFace model: {model_name}")
            if model_name == "Alibaba-NLP/gte-large-en-v1.5":
                model_kwargs["trust_remote_code"] = True
            try:
                start_time = time.time()
                embeddings = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                vector = embeddings.embed_query(self.text)
                elapsed = time.time() - start_time
                print(f"✅ Vector dim: {len(vector)}, First 5 values: {vector[:5]}")
                self.results.append([model_name, "HuggingFace", elapsed, len(vector)])
            except Exception as e:
                print(f"❌ Failed: {e}")
                self.results.append([model_name, "HuggingFace", None, "Error"])

    def test_openai(self):
        print(f"\n🔍 Testing OpenAI model: {self.OPENAI_MODEL}")
        try:
            start_time = time.time()
            embeddings = OpenAIEmbeddings(model=self.OPENAI_MODEL, openai_api_key=self.OPENAI_API_KEY)
            vector = embeddings.embed_query(self.text)
            elapsed = time.time() - start_time
            print(f"✅ Vector dim: {len(vector)}, First 5 values: {vector[:5]}")
            self.results.append([self.OPENAI_MODEL, "OpenAI", elapsed, len(vector)])
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append([self.OPENAI_MODEL, "OpenAI", None, "Error"])

    def test_mistral(self):
        print(f"\n🔍 Testing Mistral model: mistral-embed-1536")
        try:
            start_time = time.time()
            embeddings = MistralAIEmbeddings(model=self.MISTRAIL_EMBEDING_MODEL, mistral_api_key=self.MISTRAL_API_KEY)
            vector = embeddings.embed_query(self.text)
            elapsed = time.time() - start_time
            print(f"✅ Vector dim: {len(vector)}, First 5 values: {vector[:5]}")
            self.results.append([self.MISTRAIL_EMBEDING_MODEL, "Mistral", elapsed, len(vector)])
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append([self.MISTRAIL_EMBEDING_MODEL, "Mistral", None, "Error"])

    def test_groq(self):
        print(f"\n🔍 Testing Groq model: {self.GROQ_EMBEDING_MODEL}")
        try:
            start_time = time.time()
            llm = ChatGroq(
                api_key=self.GROQ_API_KEY,
                model=self.GROQ_EMBEDING_MODEL,
                temperature=0
            )
            # Use LLM output as embedding surrogate
            response = llm.invoke(self.text)
            vector = [ord(c) % 100 / 100 for c in response.content[:100]]  # Dummy embed conversion
            elapsed = time.time() - start_time
            print(f"✅ Generated pseudo-embedding, dim: {len(vector)}, First 5 values: {vector[:5]}")
            self.results.append([self.GROQ_EMBEDING_MODEL, "Groq (LLM)", elapsed, len(vector)])
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append([self.GROQ_EMBEDING_MODEL, "Groq", None, "Error"])

    def run_all_tests(self):
        self.test_huggingface_models()
        self.test_openai()
        self.test_mistral()
        self.test_groq()

    def get_results_dataframe(self):
        return pd.DataFrame(self.results, columns=["Model", "Source", "Embedding Time (s)", "Vector Dimension"])

    def print_summary(self):
        df = self.get_results_dataframe()
        print("\n📊 Summary:")
        print(df.to_string(index=False))


# Example usage
if __name__ == "__main__":
    tester = EmbeddingTester(text="AI is transforming industries.")
    tester.run_all_tests()
    df_results = tester.get_results_dataframe()
    print(df_results)
