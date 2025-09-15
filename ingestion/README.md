# 📊 Tabular File Processor with Docling + LangChain

This project provides a utility (`TabularProcessor`) to process **Excel** (`.xls`, `.xlsx`) and **CSV** files for use in **RAG (Retrieval-Augmented Generation) pipelines**.  
It uses **Docling** for document extraction and **LangChain** for text chunking.

---

## 🚀 Features
- Converts `.xls` → `.xlsx` automatically
- Converts `.csv` → `.xlsx` automatically
- Loads tabular data with **Docling**
- Splits large tables into smaller **chunks** using `RecursiveCharacterTextSplitter`
- Supports **single-file** and **batch folder** processing
- Returns **document chunks** ready for embedding or vector DB ingestion

---

## 📦 Installation

### 1. Clone this repository
```bash
git clone https://github.com/your-username/tabular-rag-processor.git
cd tabular-rag-processor
