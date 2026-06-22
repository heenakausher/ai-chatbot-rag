import os
import hashlib
import pickle
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"
import pytesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe")
import tempfile
import time
import fitz  # pip install pymupdf
from PIL import Image
import io
import streamlit as st
from dotenv import load_dotenv
import tempfile as tempfile_module
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain)

from langchain_classic.chains.retrieval import (
    create_retrieval_chain)
from collections import Counter
import re


def clean_text(text):
    watermark_patterns = [
        r"CONFIDENTIAL",
        r"DRAFT",
        r"SAMPLE",
        r"FOR INTERNAL USE ONLY",
        r"COPY"]

    for pattern in watermark_patterns:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==================================================
# APP START TIMER
# ==================================================

start = time.time()

# ==================================================
# ENV VARIABLES
# ==================================================

load_dotenv()

# ==================================================
# STREAMLIT CONFIG
# ==================================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide")

st.title("📄 PDF RAG Chatbot")

# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "current_pdf_names" not in st.session_state:
    st.session_state.current_pdf_names = []

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def save_uploaded_file(uploaded_file):
    """
    Save uploaded PDF temporarily.
    """
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())

        return temp_file.name


@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True})


@st.cache_resource
def get_llm():

    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0)



import concurrent.futures

def ocr_single_page(args):
    pdf_path, page_num = args
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[page_num]
    
    # Try direct text extraction first (works for text-based PDFs)
    direct_text = page.get_text("text").strip()
    pdf_doc.close()
    
    if len(direct_text) > 50:  # If meaningful text found, skip OCR
        clean = re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', ' ', direct_text)
        clean = re.sub(r'[^\x00-\x7F]+', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean)
        return page_num, clean_text(clean)
    
    # Fallback to OCR for image-based pages
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[page_num]
    mat = fitz.Matrix(300 / 72, 300 / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    pdf_doc.close()
    
    image = Image.open(io.BytesIO(img_bytes))
    text = pytesseract.image_to_string(image, lang="eng")
    
    # Clean non-ASCII / Arabic
    text = re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return page_num, clean_text(text)

def enrich_text(text, page_num):
    """
    Generalize enrichment for any PDF type.
    Detects patterns and boosts searchability.
    """
    enriched = text

    # Boost words inside brackets/parentheses (pronunciations, terms, codes)
    bracketed = re.findall(r'\(([A-Za-z0-9 \-]+)\)', text)
    for term in bracketed:
        term = term.strip()
        if 2 <= len(term) <= 30:  # Ignore very short or very long matches
            enriched += f"\nTerm: {term}. Definition of {term}. Meaning of {term}."

    # Boost numbered headings like #1, 1., Chapter 1, Word 1
    headings = re.findall(
        r'(?:#\s*\d+|\b(?:Chapter|Word|Section|Part|Lesson|Unit)\s+\d+|\b\d+\.)',
        text, re.IGNORECASE
    )
    for h in headings:
        enriched += f"\nSection: {h.strip()}"

    # Boost key-value patterns like "Repeats 3226 Times", "Frequency: 500"
    stats = re.findall(
        r'((?:Repeats?|Frequency|Count|Times?|Pages?)[^\n\.]{0,40})',
        text, re.IGNORECASE
    )
    for s in stats:
        enriched += f"\nStat: {s.strip()}"

    # Boost definition patterns: "X means Y", "X is defined as Y", "X: Y"
    definitions = re.findall(
        r'([A-Za-z ]{2,20}(?:\s+means?|\s+is\s+defined\s+as|\s+refers?\s+to)[^\n\.]{0,60})',
        text, re.IGNORECASE
    )
    for d in definitions:
        enriched += f"\nDefinition: {d.strip()}"

    # Boost capitalized terms (likely important labels like "Preposition", "Noun")
    caps_terms = re.findall(r'\b([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]+)*)\b', text)
    for c in set(caps_terms):
        if c not in ["Copyright", "Congratulations", "This", "After"]:
            enriched += f"\nKeyword: {c}"

    enriched += f"\nPage: {page_num + 1}"
    return enriched

def load_pdf(pdf_path):
    
    # Try fast text extraction first
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        extracted = "".join(d.page_content for d in docs).strip()
        if len(extracted) > 200:
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
            return docs
    except Exception as e:
        print(f"PyPDFLoader failed: {e}")

    # Parallel OCR fallback
    """
    Handles any PDF: text-based, image-based, or mixed.
    Per-page decision: direct text or OCR fallback.
    """
    pdf_doc = fitz.open(pdf_path)
    total_pages = len(pdf_doc)
    pdf_doc.close()

    args = [(pdf_path, i) for i in range(total_pages)]
    # Process pages in parallel (4 workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(ocr_single_page, args))
    
    docs = []
    for page_num, text in sorted(results):
        if text.strip():
            # Enrich the text with structured metadata
            enriched = enrich_text(text, page_num)
            docs.append(Document(
                page_content=enriched,
                metadata={"page": page_num + 1, "source": pdf_path, "char_count": len(text)}
            ))
    print(f"Loaded {len(docs)} pages from {pdf_path}")
    return docs


def build_vectorstore(pdf_paths):
    all_documents = []
    for pdf_path, original_name in pdf_paths:
        docs = load_pdf(pdf_path)
        for doc in docs:
            doc.metadata["source_file"] = original_name
        all_documents.extend(docs)
    print(
        f"Documents loaded: "
        f"{len(all_documents)}")
    if not all_documents:
        raise ValueError("No documents were loaded.")
    page_texts = [
        doc.page_content
        for doc in all_documents]
    counter = Counter()
    for page in page_texts:
        words = page.split()
        counter.update(set(words))

    # Find words repeated on 80%+ pages
    threshold = max(3, len(page_texts))  # must appear on every single page
    repeated_words = {
        word for word, count in counter.items()
        if count >= threshold and len(word) > 6
        and word.upper() == word}  # only ALL-CAPS words like "VIDEOPRENEUR"

    print(
        "Potential watermark words:",
        repeated_words)

    # Remove repeated watermark words
    for doc in all_documents:
        text = doc.page_content
        for word in repeated_words:
            text = text.replace(word, "")
        doc.page_content = text

    # Auto-detect average page length to set chunk size
    avg_len = sum(len(d.page_content) for d in all_documents) / len(all_documents)
    
    if avg_len < 500:
        # Short pages (flashcard/vocabulary style like Quran book)
        chunk_size = 400
        chunk_overlap = 50
    elif avg_len < 2000:
        # Medium pages (reports, articles)
        chunk_size = 800
        chunk_overlap = 100
    else:
        # Long pages (books, manuals)
        chunk_size = 1500
        chunk_overlap = 200

    print(f"Avg page length: {avg_len:.0f} chars → chunk_size={chunk_size}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            ""])

    chunks = splitter.split_documents(all_documents)
    print(
        f"Chunks created: "
        f"{len(chunks)}")
    if not chunks:
        raise ValueError("No chunks generated.")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def get_pdf_hash(pdf_paths):
    hasher = hashlib.md5()
    for path, name in pdf_paths:
        hasher.update(name.encode())
    return hasher.hexdigest()


def build_rag_chain(pdf_paths):
    cache_key = get_pdf_hash(pdf_paths)
    
    # This works on Windows, Mac, and Linux automatically
    cache_dir = tempfile_module.gettempdir()
    cache_file = os.path.join(cache_dir, f"rag_cache_{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        print("Loading from disk cache...")
        with open(cache_file, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        vectorstore = build_vectorstore(pdf_paths)
        with open(cache_file, "wb") as f:
            pickle.dump(vectorstore, f)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8,"fetch_k": 20 })

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful document assistant. Answer questions based strictly on 
    the context provided below, which is extracted from one or more PDF documents.
    Guidelines:
    - Answer only from the context. Do not guess or use outside knowledge.
    - If the question is about a word, term, or concept, find its definition, 
      meaning, type, and any stats (like frequency) mentioned in the context.
    - If the answer is not found, say: "I could not find that in the document."
    - Be concise and direct.
Context:{context}
Question: {input}
Answer:""")

    llm = get_llm()
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# ==================================================
# DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True)

    if st.session_state.current_pdf_names:

        st.success("Loaded PDFs")

        for pdf in st.session_state.current_pdf_names:
            st.write(f"📄 {pdf}")

# ==================================================
# PROCESS PDFS
# ==================================================

if uploaded_files:

    uploaded_names = sorted(
        [file.name for file in uploaded_files])

    if uploaded_names != st.session_state.current_pdf_names:

        st.session_state.messages = []

        #Show progress bar during PDF processing
        with st.spinner(""):
            progress = st.progress(0, text="Reading PDF...")
            pdf_paths_list = []
    
            for i, uploaded_file in enumerate(uploaded_files):
                pdf_path = save_uploaded_file(uploaded_file)
                pdf_paths_list.append((pdf_path, uploaded_file.name))
                progress.progress((i+1) / len(uploaded_files) / 2, 
                                 text=f"Reading {uploaded_file.name}...")
    
            progress.progress(0.6, text="Building search index...")
            st.session_state.rag_chain = build_rag_chain(pdf_paths_list)
            progress.progress(1.0, text="Done!")
            progress.empty()

        st.success(f"{len(uploaded_files)} PDF(s) processed successfully!")

# ==================================================
# CHAT INPUT
# ==================================================

user_question = st.chat_input(
    "Ask a question about the uploaded PDF(s)...")

if user_question:

    if st.session_state.rag_chain is None:

        st.warning(
            "Please upload at least one PDF.")

        st.stop()

    with st.chat_message("user"):
        st.markdown(user_question)

    st.session_state.messages.append(
        {"role": "user",
            "content": user_question})

    try:

        response = (
            st.session_state.rag_chain.invoke(
                {"input": user_question}))

        answer = response["answer"]

        source_files = set()

        for doc in response.get("context", []):

            source_files.add(
                doc.metadata.get(
                    "source_file",
                    "Unknown"))

        if source_files:

            answer += "\n\n📄 Sources:\n"

            for source in sorted(source_files):
                answer += f"- {source}\n"

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_answer = ""

            # Use streaming
            for chunk in st.session_state.rag_chain.stream({"input": user_question}):
                if "answer" in chunk:
                    full_answer += chunk["answer"]
                    placeholder.markdown(full_answer + "▌")  # typing cursor
        
            placeholder.markdown(full_answer)  # final answer

        st.session_state.messages.append(
            {"role": "assistant","content": answer})

    except Exception as e:
        st.error(f"Error: {str(e)}")

# ==================================================
# FOOTER
# ==================================================

startup_time = time.time() - start
st.sidebar.caption(
    f"⚡ Loaded in {startup_time:.2f} sec")


#Auto-detect and delete PKL cache easily
import glob

def clear_cache():
    """Delete all RAG cache files"""
    cache_dir = tempfile_module.gettempdir()
    cache_files = glob.glob(os.path.join(cache_dir, "rag_cache_*.pkl"))
    for f in cache_files:
        os.remove(f)
        print(f"Deleted cache: {f}")
    return len(cache_files)

# Add this in sidebar
with st.sidebar:
    if st.button("🗑️ Clear Cache & Reload"):
        count = clear_cache()
        st.session_state.rag_chain = None
        st.session_state.current_pdf_names = []
        st.success(f"Cleared {count} cache file(s). Please re-upload your PDF.")
        st.rerun()