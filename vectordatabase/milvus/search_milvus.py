import os
import argparse
import numpy as np
from dotenv import load_dotenv
from pymilvus import Collection, utility

# Prefer new package; fallback to community
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Reuse your existing connection helper (same folder or on PYTHONPATH)
from vd_milvus import connect_to_milvus


def _field_exists(col: Collection, name: str) -> bool:
    try:
        return any(f.name == name for f in col.schema.fields)
    except Exception:
        return False


def search_query(
    query: str,
    top_k: int = 5,
    collection_name: str | None = None,
    embed_model: str | None = None,
    metric_type: str | None = None,
    nprobe: int = 10,
) -> None:
    """Run a quick vector similarity search against Milvus and print results."""
    load_dotenv()  # harmless if vd_milvus.connect_to_milvus also loads env

    # 1) Connect (via your helper)
    connect_to_milvus()

    # 2) Resolve settings
    collection_name = collection_name or os.getenv("COLLECTION_NAME_1")
    embed_model = embed_model or os.getenv("EMBED_MODEL", "sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja")
    metric_type = metric_type or os.getenv("METRIC_TYPE", "L2")  # must match your index metric

    if not collection_name:
        print("⚠️ COLLECTION_NAME_1 is not set and no --collection was provided.")
        return
    if not utility.has_collection(collection_name):
        print(f"⚠️ Collection '{collection_name}' does not exist.")
        return

    # 3) Prepare collection and embedding model
    col = Collection(collection_name)
    col.load()

    emb = HuggingFaceEmbeddings(model_name=embed_model)
    qvec = emb.embed_query(query)
    qvec = np.asarray(qvec, dtype=np.float32).tolist()

    # 4) Search params (keep metric consistent with your index)
    params = {"metric_type": metric_type, "params": {"nprobe": int(nprobe)}}

    output_fields = ["text"]
    if _field_exists(col, "filename"):
        output_fields.append("filename")

    # 5) Run search
    results = col.search(
        data=[qvec],
        anns_field="vector",
        param=params,
        limit=int(top_k),
        output_fields=output_fields,
    )

    # 6) Pretty print (PyMilvus 2.4: don't pass a default to entity.get)
    print(f"\n🔎 Top-{top_k} results for: {query!r}\n")
    for hits in results:
        for rank, hit in enumerate(hits, start=1):
            fname = hit.entity.get("filename") if "filename" in output_fields else None
            txt_val = hit.entity.get("text")     # no default arg here
            snippet = (txt_val or "")[:400].replace("\n", " ")
            print(f"#{rank}  distance={hit.distance:.4f}  filename={fname}")
            print(f"    text: {snippet}...\n")
    print("✅ Done.\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Similarity search against Milvus (uses connection from vd_milvus.py).")
    p.add_argument("--query", "-q", help="Your search query text.")
    p.add_argument("--topk", type=int, default=5, help="Top-K results (default 5).")
    p.add_argument("--collection", "-c", default=os.getenv("COLLECTION_NAME_1"),
                   help="Collection name (default from .env or 'docling_vectors_pdf').")
    p.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja"),
                   help="HF embedding model (default from .env).")
    p.add_argument("--metric", default=os.getenv("METRIC_TYPE", "L2"),
                   help="Metric type (must match index; e.g., L2 or COSINE).")
    p.add_argument("--nprobe", type=int, default=10, help="Search nprobe (default 10).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    text = args.query or input("📝 Enter your query: ").strip()
    if not text:
        print("⚠️ Empty query. Exiting.")
    else:
        search_query(
            query=text,
            top_k=args.topk,
            collection_name=args.collection,
            embed_model=args.embed_model,
            metric_type=args.metric,
            nprobe=args.nprobe,
        )
