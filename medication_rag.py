"""
Medication Reminder Chatbot ” Prototype (ChatGroq + Chroma + Streamlit)

Fully working RAG + Groq + Chroma. No deprecated LangChain imports.
"""

import os
import zipfile
import json
import re
from typing import List

import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

# ---------- Configuration ----------
ZIP_PATH = "drug-label-0001-of-0013.json.zip"
EXTRACTED_JSON_NAME = "drug-label-0001-of-0013.json"
CHROMA_PERSIST_DIR = "./chroma_db_proto"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 50
TOP_K = 4

GROQ_API_KEY = os.environ.get("")
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL", "llama3-8b-8192")
USE_GROQ = bool(GROQ_API_KEY)

# ---------- Embeddings ----------
class HFEmbeddings:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

# ---------- Groq Wrapper ----------
def make_chatgroq():
    if not USE_GROQ:
        return None
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
    )

# ---------- ZIP Extraction ----------
def extract_json_from_zip(zip_path, expected_name):
    if os.path.exists(expected_name):
        return expected_name
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        for n in z.namelist():
            if n.endswith(".json"):
                z.extract(n)
                return n
    raise FileNotFoundError("No JSON inside zip.")

# ---------- Load Documents ----------
def load_documents_from_openfda(json_path):
    docs = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("results", data if isinstance(data, list) else [])
    
    for entry in entries:
        parts = []
        def safe_get(e, k):
            v = e.get(k)
            if v is None: return ""
            if isinstance(v, list): return "\n".join(map(str, v))
            if isinstance(v, dict): return "\n".join([f"{a}: {b}" for a,b in v.items()])
            return str(v)
        fields = ["openfda","description","indications_and_usage","dosage_and_administration","warnings","precautions"]
        for f_ in fields:
            t = safe_get(entry, f_)
            if t: parts.append(f"== {f_} ==\n{t}")
        if not parts: parts.append(json.dumps(entry))
        docs.append(Document(page_content="\n\n".join(parts), metadata={"id": entry.get("set_id", "")}))
    return docs

# ---------- Chroma ----------
def create_or_load_chroma(docs, persist_dir):
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = HFEmbeddings()
    try:
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        if getattr(vectordb, "_collection", None) and vectordb._collection.count() > 0:
            return vectordb
    except Exception:
        pass

    # Split and index
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = []
    for d in docs:
        for i, chunk in enumerate(splitter.split_text(d.page_content)):
            meta = d.metadata.copy()
            meta["chunk"] = i
            split_docs.append(Document(page_content=chunk, metadata=meta))

    vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_dir)
    try: vectordb.persist()
    except: pass
    return vectordb

# ---------- Safety ----------
RISKY_PATTERNS = [r"double dose", r"take two", r"overdose", r"diagnose", r"what should i take"]
def is_risky_query(q): return any(re.search(p, q.lower()) for p in RISKY_PATTERNS)

# ---------- RAG Answer ----------
def rag_answer(query, retriever):
    try:
        docs = retriever.invoke(query)
    except Exception:
        docs = retriever.get_relevant_documents(query)

    prompt = ["Use ONLY these passages. Include citations. No medical advice.\n"]
    for i, d in enumerate(docs):
        mid = d.metadata.get("id", "")
        prompt.append(f"[PASSAGE_{i} - id:{mid}]\n{d.page_content[:2000]}\n")
    prompt.append(f"QUESTION: {query}\nAnswer clearly with citations and say you are not a doctor.\n")
    final_prompt = "\n".join(prompt)

    llm = make_chatgroq()
    if llm is None:
        return "\n\n".join([f"PASSAGE_{i}: {d.page_content[:800]}" for i,d in enumerate(docs)])
    try:
        return llm.invoke(final_prompt)
    except Exception as e:
        return f"[LLM call failed: {e}]"

# ---------- Build Reminder JSON ----------
def build_reminder_json(input_str):
    drug, ctx = "", ""
    if "|" in input_str:
        for part in input_str.split("|"):
            if "drug:" in part.lower(): drug = part.split(":",1)[1].strip()
            if "context:" in part.lower(): ctx = part.split(":",1)[1].strip()
    prompt = f"""
Create a medication reminder JSON for '{drug}'.
Fields: drug, dosage, frequency, schedule, notes.
Context: {ctx}
Return ONLY valid JSON.
"""
    llm = make_chatgroq()
    if llm is None:
        return json.dumps({"drug": drug, "dosage": "", "frequency": "", "schedule": [], "notes": "(no LLM available)"}, indent=2)
    try: return llm.invoke(prompt)
    except Exception as e: return f"[LLM call failed: {e}]"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Medication RAG Prototype", layout="wide")
st.title("Medication Reminder Chatbot â€” Prototype (ChatGroq)")

st.sidebar.header("Setup")
if st.sidebar.button("Extract & Index"):
    try:
        st.info("Extracting JSON...")
        extracted = extract_json_from_zip(ZIP_PATH, EXTRACTED_JSON_NAME)
        st.info("Loading documents...")
        docs = load_documents_from_openfda(extracted)
        st.info("Creating Chroma index... (may take a minute)")
        vectordb = create_or_load_chroma(docs, CHROMA_PERSIST_DIR)
        st.success("Index ready.")
    except Exception as e:
        st.error(str(e))

query = st.text_area("Ask something or use 'drug: name | context: ...'")

if st.button("Submit") and query.strip():
    embeddings = HFEmbeddings()
    try:
        vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    except Exception as e:
        st.error(f"Failed to load vector DB: {e}")
        vectordb = None

    if vectordb is None:
        st.error("Vector DB not available. Please run Extract & Index first.")
    else:
        retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

        if is_risky_query(query):
            st.error("âš ï¸ Your question appears medically risky. Cannot answer.")
        elif query.lower().startswith("drug:"):
            out = build_reminder_json(query)
            try:
                parsed = json.loads(out)
                st.code(json.dumps(parsed, indent=2), language="json")
            except Exception:
                st.code(out, language="json")
        else:
            with st.spinner("Retrieving and generating answer..."):
                out = rag_answer(query, retriever)
            st.write(out)

st.markdown("---")
st.sidebar.markdown("**Run notes**")
st.sidebar.text("1) Click 'Extract & Index' once after placing the ZIP file.")
st.sidebar.text("2) Set GROQ_API_KEY in env to enable LLM answers.")
if not USE_GROQ:
    st.sidebar.warning("GROQ_API_KEY not set â€” LLM calls disabled.")

