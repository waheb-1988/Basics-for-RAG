# 🧠 Basics for RAG

This repository contains the code and resources for **Basics for RAG** from the LinkedIn article *"Mastering Chatbots with RAG: From Theory to Production"* by **PhD. Abdelouaheb Hocine**.

## 📖 Overview
This series is designed especially for students, junior data scientists, or anyone curious about how modern AI chatbots really work under the hood. Whether you're looking to understand RAG for a university project, a startup idea, or to upskill for your next job this is for you.

Each post (shared daily or weekly) will cover one key concept from indexing, embeddings, and vector databases to language models, prompt engineering, and real-world deployment. I’ll share code, examples, and tools and open the floor for questions, discussions, and collaboration in the comments.

## 📖 Literature Review: 

The concept of Retrieval-Augmented Generation (RAG) was first introduced by Patrick Lewis et al. in 2020 in their influential paper titled "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". This work proposed a novel architecture that combines a retrieval modulewhich fetches relevant documents from an external knowledge base with a generative language model to produce grounded, informative answers. By integrating retrieval directly into the generation process, RAG significantly improved performance on tasks like open domain question answering and reduced the tendency of large language models to "hallucinate" facts, marking a key advancement in knowledge-intensive NLP applications.

## 📖 Series Plan:

- **Series 1:** NLP Basics for RAG

- **Series 2:** Data Collection & Preprocessing (PDFs, Word files, CSVs)

- **Series 3:** Embeddings & Vectorization

- **Series 4:** Similarity Search

- **Series 5:** Vector Databases

- **Series 6:** LLM Models

- **Series 7:** Prompt Engineering

- **Series 8:** Orchestration & Tooling (LangChain, LlamaIndex)

- **Series 9:** Testing & Evaluation

- **Series 10:** Fine-Tuning & Optimization

- **Series 11:** Deployment & Production
---

## ⚙️ Installation

Clone this repository:

```bash
git clone https://github.com/your-username/Basics-for-RAG.git
cd Basics-for-RAG

```
## 📂 Project Structure

```
Basics-for-RAG/
│
├── ingestion/
│ ├── xsl_extractor.py # Process Excel/CSV into chunks
│ └── pdf_extractor.py # (Future) Process PDFs into chunks
│
├── embedding/
│ └── embeddings.py # Test & compare multiple embedding models (OpenAI, HuggingFace, Mistral, Groq)
│
├── data/ # Sample datasets
│ └── algeria_macro_economic_kpis.xls
│
├── notebooks/ # Jupyter/Colab exploration
│ │
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```
## Install dependencies

```bash

pip install -r requirements.txt
```
