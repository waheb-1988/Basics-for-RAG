import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from dotenv import load_dotenv

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

# Docling (multi-format: pdf, xlsx/xls, csv, docx, pptx, etc.)
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

# Embeddings (prefer new pkg; fallback to community)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings


# ===================== Utilities =====================

def as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    val = os.getenv(name, default)
    if required and not val:
        raise ValueError(f"Missing required env var: {name}")
    return val


def field_exists(col: Collection, field_name: str) -> bool:
    return any(f.name == field_name for f in col.schema.fields)


# ===================== Milvus =====================

def connect_to_milvus() -> None:
    """
    Connect to Milvus (local or cloud). Prefers TOKEN for Zilliz Cloud.
    ENV options:
      USE_LOCAL=false
      LOCAL_MILVUS_URI=http://localhost:19530
      MILVUS_URI=...
      MILVUS_TOKEN=project_id:api_key   (preferred)
      MILVUS_USER=... ; MILVUS_PASSWORD=... (fallback)
    """
    use_local = as_bool(os.getenv("USE_LOCAL", "false"))
    if use_local:
        uri = get_env("LOCAL_MILVUS_URI", "http://localhost:19530")
        connections.connect(alias="default", uri=uri)
        print("✅ Connected to Milvus (Local).")
        return

    uri = get_env("MILVUS_URI", required=True)
    if "<your-zilliz-or-milvus-uri>" in uri:
        raise ValueError("MILVUS_URI still contains a placeholder. Update your .env or pass --milvus-uri.")

    token = os.getenv("MILVUS_TOKEN")
    user = os.getenv("MILVUS_USER")
    password = os.getenv("MILVUS_PASSWORD")

    if token:
        connections.connect(alias="default", uri=uri, token=token)
    else:
        connections.connect(alias="default", uri=uri, user=user, password=password)

    print(f"✅ Connected to Milvus (Cloud) at {uri}.")


def _vector_dim_from_collection(col: Collection, field_name: str = "vector") -> int:
    f = next(f for f in col.schema.fields if f.name == field_name)
    return int(f.params.get("dim", 0))


def ensure_collection(
    collection_name: str,
    vector_dim: int,
    *,
    metric_type: str = "L2",
    index_type: str = "IVF_FLAT",
    nlist: int = 128,
) -> Collection:
    """
    Create collection if absent; validate dim if present.
    Creates index if missing. Returns an *unloaded* collection (we load after inserts).
    Schema fields:
      - id (auto)
      - vector (FLOAT_VECTOR)
      - text (VARCHAR)
      - filename (VARCHAR)  <-- used for delete-by-filename and provenance
    """
    if utility.has_collection(collection_name):
        col = Collection(collection_name)
        schema_dim = _vector_dim_from_collection(col, "vector")
        if schema_dim != vector_dim:
            raise ValueError(
                f"Collection '{collection_name}' dim={schema_dim} ≠ embedding dim={vector_dim}. "
                f"Create a new collection or use a matching embedding model."
            )
        print(f"✅ Collection '{collection_name}' exists (dim={schema_dim}).")
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
        ]
        schema = CollectionSchema(fields, description="Docling document embeddings (with filename)")
        col = Collection(name=collection_name, schema=schema)
        print(f"✅ Created collection '{collection_name}' (dim={vector_dim}).")

    # Create index if not present (idempotent)
    if not getattr(col, "indexes", None):
        index_params = {"index_type": index_type, "metric_type": metric_type, "params": {"nlist": int(nlist)}}
        col.create_index("vector", index_params)
        print(f"✅ Created index on 'vector' with {index_params}.")

    return col


# ===================== Docling Loading =====================

def build_chunker(max_tokens: int, min_tokens: int, overlap: int) -> HybridChunker:
    """Configure HybridChunker."""
    return HybridChunker(
        tokenizer="bert-base-uncased",
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap=overlap,
    )


def load_with_docling(file_path: str, chunker: HybridChunker):
    """
    Uses DoclingLoader across many file types (pdf, xlsx/xls, csv, docx, pptx, etc.).
    Returns a list of LangChain docs (each having .page_content).
    """
    loader = DoclingLoader(
        file_path=file_path,
        export_type=ExportType.DOC_CHUNKS,
        chunker=chunker,
    )
    docs = loader.load()
    print(f"✅ Loaded and chunked file: {file_path} (chunks={len(docs)})")
    return docs


# ===================== Embeddings =====================

def build_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def detect_embedding_dim(embedding_model: HuggingFaceEmbeddings) -> int:
    return len(embedding_model.embed_query("dimension probe"))


def embed_docs(docs, embedding_model: HuggingFaceEmbeddings) -> Tuple[List[List[float]], List[str]]:
    vectors, texts = [], []
    for d in docs:
        try:
            v = embedding_model.embed_query(d.page_content)
            vectors.append(v)
            texts.append(d.page_content)
        except Exception as e:
            print(f"❌ Error embedding chunk: {e}")
    print(f"✅ Embedded {len(vectors)} chunks.")
    return vectors, texts


# ===================== Insert / Search / Maintenance =====================

def insert_records(collection: Collection, vectors: List[List[float]], texts: List[str], filenames: Optional[List[str]] = None) -> int:
    """Column-wise insert. If 'filename' field exists, include it."""
    if not vectors:
        print("❌ No vectors to insert.")
        return 0

    vectors = [np.asarray(v, dtype=np.float32).tolist() for v in vectors]

    expected_dim = _vector_dim_from_collection(collection, "vector")
    actual_dim = len(vectors[0])
    if actual_dim != expected_dim:
        raise ValueError(f"Vector dim mismatch: expected {expected_dim}, got {actual_dim}")

    payload: List = [vectors, texts]
    if field_exists(collection, "filename"):
        if filenames is None:
            filenames = [""] * len(texts)
        payload.append(filenames)

    res = collection.insert(payload)
    rows = getattr(res, "insert_count", len(vectors))
    print(f"✅ Inserted {rows} vectors.")
    return rows


def search_collection(
    collection_name: str,
    embedding_model: HuggingFaceEmbeddings,
    query_text: str,
    top_k: int = 3,
    metric_type: str = "L2",
    nprobe: int = 10,
):
    """Simple semantic search tester."""
    connect_to_milvus()
    if not utility.has_collection(collection_name):
        print(f"⚠️ Collection '{collection_name}' does not exist.")
        return

    col = Collection(collection_name)
    col.load()

    qvec = embedding_model.embed_query(query_text)
    params = {"metric_type": metric_type, "params": {"nprobe": nprobe}}

    output_fields = ["text"]
    if field_exists(col, "filename"):
        output_fields.append("filename")

    results = col.search(
        data=[np.asarray(qvec, dtype=np.float32).tolist()],
        anns_field="vector",
        param=params,
        limit=top_k,
        output_fields=output_fields,
    )

    print(f"\n🔎 Top-{top_k} results for: {query_text!r}")
    for i, hits in enumerate(results):
        for rk, hit in enumerate(hits):
            fname = hit.entity.get("filename") if "filename" in output_fields else None
            txt = hit.entity.get("text", "")[:200].replace("\n", " ")
            print(f"#{rk+1}  distance={hit.distance:.4f}  filename={fname}")
            print(f"     text: {txt}...")
    print("")


def delete_file_from_collection(collection_name: str, filename_keyword: str):
    """
    Deletes all vector entries related to a specific file by 'filename' (no extension required).
    Requires 'filename' field in schema.
    """
    try:
        connect_to_milvus()
        if not utility.has_collection(collection_name):
            print(f"⚠️ Collection '{collection_name}' does not exist.")
            return
        collection = Collection(collection_name)
        if not field_exists(collection, "filename"):
            print("⚠️ Collection has no 'filename' field. Re-ingest into a collection that includes it.")
            return

        collection.load()
        expr = f'filename == "{filename_keyword}"'
        print(f"🗑️ Deleting entries where: {expr}")
        collection.delete(expr)
        collection.flush()
        print(f"✅ Deleted all chunks with filename '{filename_keyword}' in '{collection_name}'")
    except Exception as e:
        print(f"❌ Error deleting from collection: {e}")


def delete_collection_by_name(collection_name: str):
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


def add_file_to_existing_collection(
    file_path: str,
    collection_name: str,
    embedding_model: HuggingFaceEmbeddings,
    max_tokens: int = 500,
    min_tokens: int = 200,
    overlap: int = 50,
):
    """Add a single file into an existing collection (uses Docling for any supported type)."""
    try:
        if not Path(file_path).is_file():
            raise ValueError(f"File path {file_path} is not a valid file.")

        connect_to_milvus()
        if not utility.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        col = Collection(collection_name)
        # Validate dim
        expected_dim = _vector_dim_from_collection(col)
        actual_dim = detect_embedding_dim(embedding_model)
        if expected_dim != actual_dim:
            raise ValueError(
                f"Embedding dim ({actual_dim}) does not match collection dim ({expected_dim}). "
                "Use the same embedding model used to create the collection."
            )

        chunker = build_chunker(max_tokens, min_tokens, overlap)
        docs = load_with_docling(file_path, chunker)
        vectors, texts = embed_docs(docs, embedding_model)

        fname = Path(file_path).stem
        filenames = [fname] * len(texts)

        inserted = insert_records(col, vectors, texts, filenames=filenames)
        print(f"✅ File '{file_path}' added ({inserted} chunks).")
    except Exception as e:
        print(f"❌ Error adding file to collection: {e}")


# ===================== File Gathering =====================

def list_files(base_path: str, extension: str, recursive: bool) -> List[Path]:
    """
    Collect files with given extension from base_path.
    - If base_path is a file -> return [file] if it matches extension.
    - If base_path is a folder -> return all files matching extension (optionally recursive).
    """
    base = Path(base_path)
    ext = extension.lower().lstrip(".")

    if base.is_file():
        return [base] if base.suffix.lower().lstrip(".") == ext else []

    if base.is_dir():
        pattern = f"**/*.{ext}" if recursive else f"*.{ext}"
        return sorted(base.glob(pattern))

    return []


# ===================== CLI / Main =====================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest & manage Docling-supported files (pdf/xlsx/xls/csv/docx/pptx/...) in Milvus."
    )

    # --- bulk control flags ---
    p.add_argument("--ingest", action="store_true",
                   help="Enable bulk ingestion using --path or DOC_BASE_PATH (env).")
    p.add_argument("--no-bulk", action="store_true",
                   help="Disable bulk ingestion for this run (overrides DOC_BASE_PATH).")

    # Common options (env fallbacks)
    p.add_argument("--collection", "-c", default=os.getenv("COLLECTION_NAME_1", "docling_vectors"),
                   help="Milvus collection name.")
    p.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja"),
                   help="HF embedding model.")
    p.add_argument("--metric", default=os.getenv("METRIC_TYPE", "L2"),
                   help="Milvus metric type (e.g., L2 or COSINE).")
    p.add_argument("--index-type", default=os.getenv("INDEX_TYPE", "IVF_FLAT"),
                   help="Index type (FLAT, IVF_FLAT, HNSW, IVF_PQ...).")
    p.add_argument("--nlist", type=int, default=int(os.getenv("INDEX_NLIST", "128")),
                   help="Index nlist (for IVF_*).")
    p.add_argument("--max-tokens", type=int, default=int(os.getenv("CHUNK_MAX_TOKENS", "500")),
                   help="Chunker max tokens (default 500).")
    p.add_argument("--min-tokens", type=int, default=int(os.getenv("CHUNK_MIN_TOKENS", "200")),
                   help="Chunker min tokens (default 200).")
    p.add_argument("--overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "50")),
                   help="Chunker token overlap (default 50).")

    # Bulk controls (path/ext)
    p.add_argument("--path", "-p", default=os.getenv("DOC_BASE_PATH"),
                   help="Folder or file path. Defaults to DOC_BASE_PATH from .env.")
    p.add_argument("--ext", "-e", default=os.getenv("DOC_EXTENSION", "pdf"),
                   help="File extension filter (e.g., pdf/xlsx/csv). Defaults to DOC_EXTENSION from .env.")
    p.add_argument("--recursive", "-r", action="store_true",
                   help="Recurse into subfolders when --path is a directory.")

    # Single-file add
    p.add_argument("--file", help="Add a single file to the collection (bypasses bulk unless --ingest).")

    # Maintenance
    p.add_argument("--delete-file", help="Delete all chunks whose filename equals this value (no extension).")
    p.add_argument("--delete-collection", help="Delete the given collection name and all data.")
    p.add_argument("--query", help="Run a quick vector search with this text.")
    p.add_argument("--topk", type=int, default=3, help="Top-K results for search (default 3).")
    p.add_argument("--nprobe", type=int, default=10, help="Search nprobe (default 10).")

    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    # Did the user explicitly pass --path/-p on CLI?
    user_explicit_path = any(tok in ("--path", "-p") for tok in sys.argv)

    # Build embedding model only when needed
    embedding_model: Optional[HuggingFaceEmbeddings] = None
    if any([args.query, args.file, args.ingest, user_explicit_path]):
        embedding_model = build_embedding_model(args.embed_model)

    # 1) Delete collection (short-circuit unless also ingesting)
    if args.delete_collection:
        delete_collection_by_name(args.delete_collection)
        if not args.ingest:
            return

    # 2) Delete by filename (short-circuit unless also ingesting)
    if args.delete_file:
        delete_file_from_collection(args.collection, args.delete_file)
        if not args.ingest:
            return

    # 3) Search (short-circuit unless also ingesting or adding file)
    if args.query:
        if embedding_model is None:
            embedding_model = build_embedding_model(args.embed_model)
        search_collection(
            collection_name=args.collection,
            embedding_model=embedding_model,
            query_text=args.query,
            top_k=args.topk,
            metric_type=args.metric,
            nprobe=args.nprobe,
        )
        if not args.ingest and not args.file:
            return

    # 4) Add a single file (bypasses bulk unless --ingest specified)
    if args.file:
        if embedding_model is None:
            embedding_model = build_embedding_model(args.embed_model)
        add_file_to_existing_collection(
            file_path=args.file,
            collection_name=args.collection,
            embedding_model=embedding_model,
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
            overlap=args.overlap,
        )
        if not args.ingest:
            return

    # 5) Bulk ingestion:
    # - requires args.ingest AND a path (either from CLI or .env)
    # - respects --no-bulk
    do_bulk = bool(args.path) and args.ingest and not args.no_bulk

    if do_bulk:
        base_path = str(Path(args.path).expanduser())
        extension = args.ext

        # Connect
        connect_to_milvus()

        if embedding_model is None:
            embedding_model = build_embedding_model(args.embed_model)
        actual_dim = detect_embedding_dim(embedding_model)
        print(f"ℹ️ Detected embedding dimension: {actual_dim} (model='{args.embed_model}')")

        collection = ensure_collection(
            collection_name=args.collection,
            vector_dim=actual_dim,
            metric_type=args.metric,
            index_type=args.index_type,
            nlist=args.nlist,
        )

        files = list_files(base_path, extension, recursive=args.recursive)
        if not files:
            print(f"❌ No .{extension} files found at {base_path} (recursive={args.recursive})")
            sys.exit(1)

        print(f"📂 Found {len(files)} file(s) with extension .{extension}")

        total_inserted = 0
        for f in files:
            print(f"\n--- Processing {f} ---")
            chunker = build_chunker(args.max_tokens, args.min_tokens, args.overlap)
            docs = load_with_docling(str(f), chunker)
            vectors, texts = embed_docs(docs, embedding_model)
            filenames = [Path(f).stem] * len(texts)
            inserted = insert_records(collection, vectors, texts, filenames=filenames)
            total_inserted += inserted
            if inserted > 0:
                print(f"✅ Inserted {inserted} chunks from {f}")

        if total_inserted > 0:
            collection.load()
        print(f"\n🎉 Done. Inserted {total_inserted} chunks total. Collection '{args.collection}' is ready for search.")
        return

    # 6) If nothing else was requested
    if not any([args.file, args.query, args.delete_file, args.delete_collection, do_bulk]):
        print("ℹ️ Nothing to do. Use --file to add one file, --query to search, "
              "--delete-file to remove by filename, --delete-collection to drop a collection, "
              "or --ingest with --path/--ext to bulk ingest.")


if __name__ == "__main__":
    main()
