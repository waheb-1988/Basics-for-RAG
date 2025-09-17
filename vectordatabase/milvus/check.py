from vd_milvus import connect_to_milvus
from pymilvus import Collection
import os
from dotenv import load_dotenv
load_dotenv()  # <— ensure .env is loaded even in one-liners/imports
connect_to_milvus()
col = Collection(os.getenv("COLLECTION_NAME_1", "docling_vectors"))
col.load()
rows = col.query(expr='filename == "algeria_macro_economic_kpis"', output_fields=["id"])
print("Chunks for file:", len(rows))
