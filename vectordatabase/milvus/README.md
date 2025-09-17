# Milvus Setup & Ingestion Guide (Docling + LangChain)

This README shows how to:

1. Select a **data folder** and **file extension** for bulk ingestion
2. Create a **managed Milvus (Zilliz Cloud)** account
3. Run **Milvus in the cloud** and connect from Python
4. Run **Milvus locally with Docker** (all commands)

> The ingestion script supports any Docling‑supported file type: **.pdf**, **.xlsx/.xls**, **.csv**, **.docx**, **.pptx**, etc. It chunks content, embeds it with a HuggingFace model, and inserts vectors + text into Milvus.

---

## 1) Select the data path and extension

You can control which files get ingested in two ways: **CLI flags** or **.env** variables. The script finds files by **extension** inside a given **folder** (or ingests a single file if you pass one).

### Option A — CLI flags (recommended for quick runs)
```powershell
# Ingest all PDFs in a folder (non-recursive)
python .\milvus_handler_xsl.py --path "C:\data\pdfs" --ext pdf

# Recurse into subfolders
python .\milvus_handler_xsl.py -p "C:\data\pdfs" -e pdf --recursive

# Excel and CSV examples
python .\milvus_handler_xsl.py -p "C:\data\excels" -e xlsx
python .\milvus_handler_xsl.py -p "C:\data\csvs" -e csv

# Control chunking for speed/quality (tokens)
python .\milvus_handler_xsl.py -p "C:\data\pdfs" -e pdf --max-tokens 500 --overlap 50
```

### Option B — `.env`
Create a `.env` next to your script (the script auto-loads it):
```env
DOC_BASE_PATH=C:\data\pdfs
DOC_EXTENSION=pdf
CHUNK_MAX_TOKENS=500
CHUNK_OVERLAP=50
```
Then simply run:
```powershell
python .\milvus_handler_xsl.py
```

---

## 2) Create a managed Milvus (Zilliz Cloud) account

1. Sign up / sign in to **Zilliz Cloud** (managed Milvus).
2. **Create a Serverless cluster** (choose a region; defaults are fine for testing).
3. Wait until the cluster status is **RUNNING**, then open the **Cluster Details** page.
4. On the **Connect** card, copy the **cluster public endpoint** (URI).
5. Create an **API key** (or use cluster credentials). Copy the value — this is your **token**.
6. Save these into your `.env` (preferred) or pass via CLI/CI:
```env
USE_LOCAL=false
MILVUS_URI=https://<your-cluster-endpoint>
MILVUS_TOKEN=<project_id:api_key_or_username:password>
COLLECTION_NAME_1=docling_vectors
```

**Quick Python connection test**
```python
from pymilvus import connections, utility
connections.connect(alias="default", uri="https://<your-cluster-endpoint>", token="<your_token>")
print("Server:", utility.get_server_version())
```

> You can also use `user=` and `password=` instead of a `token`, but a token (API key) is simplest.

---

## 3) Run Milvus in the cloud (and ingest)

Once your cluster is ready and you have `MILVUS_URI` and `MILVUS_TOKEN` set, you can ingest data immediately.

**Example: ingest PDFs with larger chunks for speed**
```powershell
python .\milvus_handler_xsl.py -p "C:\data\pdfs" -e pdf --max-tokens 500 --overlap 50 `
  --collection docling_vectors --metric L2 --index-type IVF_FLAT --nlist 128
```

If everything is set correctly, the script will:

- Auto‑detect the embedding dimension
- Create/validate the collection and index
- Chunk, embed, and insert your documents

---

## 4) Run Milvus locally with Docker (standalone)

> Works on Windows (Docker Desktop), macOS, and Linux. Below are commands for both Linux/macOS and Windows PowerShell.

### A) Download the official Docker Compose file

**Linux/macOS:**

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.4.23/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

**Windows PowerShell:**

```powershell
curl.exe -L -o docker-compose.yml `
  https://github.com/milvus-io/milvus/releases/download/v2.4.23/milvus-standalone-docker-compose.yml
```

### B) Start Milvus

```bash
# Linux/macOS
sudo docker compose up -d
```

```powershell
# Windows PowerShell
docker compose up -d
```

This brings up three containers: `milvus-standalone`, `milvus-minio`, and `milvus-etcd`. By default, Milvus listens on **19530**.

### C) Check status / stop / clean up

```bash
# Status (Linux/macOS syntax works on Windows PowerShell too)
docker compose ps

# Stop
docker compose down

# Remove local volumes (data)
# Linux/macOS
rm -rf volumes
# Windows PowerShell
Remove-Item -Recurse -Force volumes
```

### D) Connect to local Milvus from Python

```python
from pymilvus import connections
connections.connect(uri="http://localhost:19530")
```

### E) Local `.env` sample

```env
USE_LOCAL=true
LOCAL_MILVUS_URI=http://localhost:19530
COLLECTION_NAME_1=docling_vectors
```

---

## Windows model-cache note (symlinks)

On Windows, when Docling/Hugging Face downloads models the first time, you may hit **WinError 1314** (symlink privilege). Fix it by doing **one** of:

- Run **PowerShell as Administrator** once to allow model download, then run normally later.
- **Enable Developer Mode**: Settings → Privacy & security → For developers → Developer Mode → On.
- Or pre-download models / avoid symlinks (optional):
  - Upgrade hub: `pip install -U huggingface_hub`
  - (Optional) Set a clean cache: `setx HF_HOME C:\hf_cache` and open a new shell.

---

## Script reference

**Install dependencies (once):**
```bash
pip install pymilvus==2.4.* python-dotenv langchain-community docling langchain-docling numpy
```

# --- Ingestion & management (vd_milvus.py) ---

# PDFs non-recursive (BULK): requires --ingest
python .\vd_milvus.py --ingest --path "C:\data\pdfs" --ext pdf

# With recursion (BULK)
python .\vd_milvus.py --ingest -p "C:\data\pdfs" -e pdf --recursive

python .\vd_milvus.py --ingest -p "C:\Abdelouaheb\perso\Data_science_2024_projects\2025\Basics-for-RAG\data\pdf" -e pdf --recursive 

# Use bigger chunks for faster tests (BULK)
python .\vd_milvus.py --ingest -p "C:\data\pdfs" -e pdf --max-tokens 500 --overlap 50

# Excel / CSV (BULK)
python .\vd_milvus.py --ingest -p "C:\data\excels" -e xlsx
python .\vd_milvus.py --ingest -p "C:\data\csvs"   -e csv

# Add a single file ONLY (no bulk)
python .\vd_milvus.py --file "C:\data\one\report.pdf"

python .\vd_milvus.py --file 
"C:\Abdelouaheb\perso\Data_science_2024_projects\2025\Basics-for-RAG\data\pdfadd\KSP_Paper_Award_Fall_2014_CIGAINERO_Jacob.pdf"
python .\vd_milvus.py --file  "C:\Abdelouaheb\perso\Data_science_2024_projects\2025\Basics-for-RAG\data\xlsx\algeria_macro_economic_kpis.xlsx"
# Add a single file AND then bulk ingest PDFs
python .\vd_milvus.py --file "C:\data\one\report.pdf" --ingest -p "C:\data\pdfs" -e pdf

# Delete by filename (no extension) ONLY
python .\vd_milvus.py --delete-file report
python .\vd_milvus.py --delete-file "KSP_Paper_Award_Fall_2014_CIGAINERO_Jacob"
python .\vd_milvus.py --delete-file "algeria_macro_economic_kpis"
# Delete by filename and then bulk ingest (explicit)
python .\vd_milvus.py --delete-file report --ingest

# Hard kill-switch for any bulk this run
python .\vd_milvus.py --delete-file report --no-bulk

# Drop the whole collection
python .\vd_milvus.py --delete-collection docling_vectors


# --- Separate similarity search script (search_milvus.py) ---

# Interactive prompt
python .\search_milvus.py

# One-liner with query
python .\search_milvus.py --query "inflation forecast for Algeria" --topk 5

# If your index uses COSINE
python .\search_milvus.py -q "GDP growth outlook" --metric COSINE --nprobe 16

# Target a specific collection
python .\search_milvus.py -q "manufacturing PMI" -c docling_vectors_pdf


**Troubleshooting tips:**

- **Connection failed (localhost)** → Make sure Docker is running and `docker compose up -d` succeeded.
- **Connection failed (cloud)** → Check `MILVUS_URI`, token or user/password; ensure the cluster is RUNNING.
- **Vector dimension mismatch** → Use the embedding model that matches the collection dim, or let the script create a new collection after auto‑detecting the dimension.
- **DataNotMatchException on insert** → The script uses **column‑wise** insert `[vectors, texts]`; don’t switch to row‑wise unless you change the shape accordingly.
- **Sequence length > 512 warning** → Lower `--max-tokens` to 480 (or keep 500 for quick tests; the model may truncate slightly).
