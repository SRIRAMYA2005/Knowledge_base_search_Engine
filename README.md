# Local RAG PDF Summarizer

This project is a **Knowledge-Based Search Engine using LLM (RAG - Retrieval Augmented Generation)** built with **Python**, **FastAPI**, and **Hugging Face models**.  
It allows users to **upload PDF files**, **ingest them into a local vector database (FAISS)**, and **generate concise summaries or answer questions** using AI — all running **completely locally** (no external API keys required).

---

## Features

- **PDF Upload & Ingestion** – Read and split PDFs into chunks for better processing.  
- **Embedding & Storage** – Uses `all-MiniLM-L6-v2` to generate embeddings and stores them in FAISS.  
- **RAG Pipeline** – Retrieve relevant chunks and generate context-aware summaries using `flan-t5-base`.  
- **FastAPI Backend** – Clean and efficient REST API to handle summarization and queries.  
- **Browser Interface** – Simple frontend using HTML & CSS for uploading PDFs and viewing summaries.  
- **Fully Local** – No API keys or external services needed.

---

## Project Structure
Knowledge_base-search-Engine/
│
├── backend/
│ ├── app.py # FastAPI server with summarization & query endpoints
│ ├── ingest.py # PDF ingestion and embedding generation script
│ ├── index.faiss # Vector database storing embeddings
│ ├── meta.pkl # Metadata (chunk sources)
│
├── frontend/
│ ├── index.html # Frontend HTML page for uploading and summarizing PDFs
│ ├── style.css # Styling for the frontend page
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Ignored files (e.g., venv, cache)

## Tech Stack

- **Language**: Python 3.10+
- **Framework**: FastAPI
- **Vector Store**: FAISS
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **LLM**: Hugging Face `flan-t5-base`
- **Frontend**: HTML + CSS (No JS required)
- **Server**: Uvicorn


## Installation

### Clone the Repository
bash
https://github.com/SRIRAMYA2005/Knowledge_base_search_Engine.git
cd Knowledge_base-search-Engine


