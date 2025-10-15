# backend/ingest.py
import os, glob, pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pdfplumber
from tqdm import tqdm

DATA_DIR = "./data"
INDEX_PATH = "./index.faiss"
META_PATH = "./meta.pkl"

EMB_MODEL = "all-MiniLM-L6-v2"   # small, fast
CHUNK_SIZE = 1000
OVERLAP = 200
MAX_FILE_SIZE_MB = 50  # skip absurdly large files

def pdf_chunks(path, max_chars=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            buffer = ""
            for page in pdf.pages:
                text = page.extract_text() or ""
                buffer += text + "\n"
                while len(buffer) >= max_chars:
                    chunks.append(buffer[:max_chars])
                    buffer = buffer[max_chars - overlap:]
            if buffer.strip():
                chunks.append(buffer)
    except Exception as e:
        print(f"Warning: failed to read PDF {path}: {e}")
    return chunks

def txt_chunks(path, max_chars=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    buffer = ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                buffer += line
                while len(buffer) >= max_chars:
                    chunks.append(buffer[:max_chars])
                    buffer = buffer[max_chars - overlap:]
        if buffer.strip():
            chunks.append(buffer)
    except Exception as e:
        print(f"Warning: failed to read TXT {path}: {e}")
    return chunks

def ingest_folder(folder=DATA_DIR):
    model = SentenceTransformer(EMB_MODEL)
    docs_meta = []  # list of dicts: {id, source, text}
    texts = []

    files = sorted(glob.glob(os.path.join(folder, "*")))
    for path in files:
        if not path.lower().endswith((".pdf", ".txt")):
            continue
        basename = os.path.basename(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            print(f"Skipping {basename} ({size_mb:.2f} MB, too large)")
            continue

        print("Processing", basename)
        if path.lower().endswith(".pdf"):
            chunks = pdf_chunks(path)
        else:
            chunks = txt_chunks(path)

        if not chunks:
            print(f"No chunks for {basename}, skipping.")
            continue

        for i, c in enumerate(chunks):
            doc_id = f"{basename}_{i}"
            docs_meta.append({"id": doc_id, "source": basename, "text": c})
            texts.append(c)

    if not texts:
        print("No text found - nothing to index.")
        return

    print(f"Encoding {len(texts)} chunks with {EMB_MODEL}...")
    embeds = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeds.shape[1]

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeds.astype('float32'))

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(docs_meta, f)

    print(f"Saved index -> {INDEX_PATH}, meta -> {META_PATH}")

if __name__ == "__main__":
    ingest_folder()
