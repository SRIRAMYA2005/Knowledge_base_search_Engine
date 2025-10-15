# backend/app.py
import os, pickle, asyncio
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, HTMLResponse
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import torch

# ---------------------------
# Config
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE_DIR, "index.faiss")
META_PATH = os.path.join(BASE_DIR, "meta.pkl")

EMB_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"  
TOP_K = 4

# ---------------------------
# Init App + Load Models
# ---------------------------
app = FastAPI(title="Local RAG API")

# Embedding model
embed_model = SentenceTransformer(EMB_MODEL)

# FAISS index + metadata
if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    raise FileNotFoundError("index.faiss or meta.pkl not found. Run ingest.py first.")

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    METADATA = pickle.load(f)

# LLM pipeline
device = 0 if torch.cuda.is_available() else -1
qa_pipeline = pipeline("text2text-generation", model=LLM_MODEL, device=device)

# ---------------------------
# Helper functions
# ---------------------------
def retrieve_docs(query: str, top_k: int = TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb.astype("float32"), top_k)
    hits = []

    for idx in I[0]:
        if idx < 0 or idx >= len(METADATA):
            continue
        hits.append(METADATA[idx])
    return hits

def make_rag_prompt(question: str, contexts: list):
    ctx_text = "\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in contexts])
    return (
        f"Use the context below to answer the question. Be concise and cite sources.\n\n"
        f"Context:\n{ctx_text}\n\nQuestion: {question}\n\nAnswer:"
    )

def generate_with_pipeline(prompt: str, max_length: int = 512):
    out = qa_pipeline(prompt, max_length=max_length, do_sample=False)
    return out[0].get("generated_text", "").strip()

async def summarize_chunks(chunks):
    """Summarize multiple PDF chunks and merge them."""
    loop = asyncio.get_event_loop()
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        text = chunk["text"].strip()
        if not text:
            continue

        prompt = (
            f"Summarize the following text clearly and concisely in 3-5 sentences:\n\n"
            f"{text}\n\nSummary:"
        )
        summary = await loop.run_in_executor(None, generate_with_pipeline, prompt)
        chunk_summaries.append(f"Chunk {i+1}: {summary}")

    if not chunk_summaries:
        return "No content found to summarize."

    # Combine the chunk summaries into one final summary
    final_prompt = (
        "Combine the following summaries into a single clear and structured overall summary. "
        "Do not repeat points. Focus on key ideas and important information.\n\n"
        + "\n\n".join(chunk_summaries)
        + "\n\nOverall Summary:"
    )
    final_summary = await loop.run_in_executor(None, generate_with_pipeline, final_prompt)
    return final_summary

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
async def root():
    html = """
    <html>
    <head>
    <title>Local RAG PDF Summarizer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(to right, #e0f7fa, #b2ebf2);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background: #ffffff;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            width: 600px;
            max-width: 90%;
            text-align: center;
        }
        h2 {
            color: #007BFF;
            margin-bottom: 25px;
        }
        input[type="text"], input[type="file"] {
            width: 100%;
            padding: 12px 20px;
            margin: 10px 0;
            box-sizing: border-box;
            border: 2px solid #007BFF;
            border-radius: 8px;
        }
        input[type="submit"] {
            background-color: #007BFF;
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
        }
        input[type="submit"]:hover {
            background-color: #0056b3;
        }
        .summary {
            margin-top: 20px;
            padding: 20px;
            background: #e0f2f1;
            border-left: 5px solid #00796b;
            border-radius: 8px;
            color: #004d40;
            white-space: pre-wrap;
            text-align: left;
            max-height: 400px;       /* Scrollable height */
            overflow-y: auto;        /* Adds vertical scroll if content exceeds */
        }
        a.back-link {
            display: inline-block;
            margin-top: 20px;
            text-decoration: none;
            color: #007BFF;
        }
        a.back-link:hover {
            color: #0056b3;
        }
    </style>
    </head>
    <body>
      <div class="container">
        <h2>Summarize a PDF (Local RAG)</h2>
        <form action="/summarize-browser" method="post">
          <input name="filename" placeholder="Enter filename (e.g. mydoc.pdf)" size="50">
          <input type="submit" value="Summarize">
        </form>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(html)

@app.post("/summarize-browser", response_class=HTMLResponse)
async def summarize_browser(filename: str = Form(...)):
    try:
        chunks = [m for m in METADATA if m["source"] == filename]
        if not chunks:
            content = f"No chunks found for {filename}"
        else:
            content = await summarize_chunks(chunks)
    except Exception as e:
        content = f"Error: {e}"

    return f"""
    <html>
    <head>
    <title>Summary for {filename}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: linear-gradient(to right, #e0f7fa, #b2ebf2);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }}
        .container {{
            background: #ffffff;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            width: 600px;
            max-width: 90%;
            text-align: center;
        }}
        h2 {{
            color: #007BFF;
            margin-bottom: 25px;
        }}
        .summary {{
            margin-top: 20px;
            padding: 20px;
            background: #e0f2f1;
            border-left: 5px solid #00796b;
            border-radius: 8px;
            color: #004d40;
            white-space: pre-wrap;
            text-align: left;
            max-height: 400px;
            overflow-y: auto;
        }}
        a.back-link {{
            display: inline-block;
            margin-top: 20px;
            text-decoration: none;
            color: #007BFF;
        }}
        a.back-link:hover {{
            color: #0056b3;
        }}
    </style>
    </head>
    <body>
      <div class="container">
        <h2>Summary for {filename}</h2>
        <div class="summary">{content}</div>
        <a class="back-link" href='/'>Back</a>
      </div>
    </body>
    </html>
    """
