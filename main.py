# main.py
import os
import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
# from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Try import of Ollama wrapper (langchain-ollama)
try:
    from langchain_ollama import OllamaLLM
except Exception:
    from langchain_ollama.llms import OllamaLLM

st.set_page_config(page_title="Local RAG — Ollama", layout="wide")
st.title("📚 Local RAG Chatbot")

# Config — can set env vars or edit here
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
# model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract selectable text from PDF using PyMuPDF (fitz)."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF bytes: {e}")
    texts = []
    for page in doc:
        texts.append(page.get_text() or "")
    return "\n\n".join(texts)

def build_vectordb_from_text(text: str, persist_dir: str = CHROMA_DIR):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.create_documents([text])
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings,
                                     persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

def create_qa_chain_from_chroma(vectordb):
    # Use Ollama via langchain-ollama Integration
    llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.0)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa

# Upload UI
uploaded = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt", "md"])
if uploaded:
    st.write("Uploaded:", uploaded.name)
    data = uploaded.read()
    if uploaded.type == "application/pdf":
        extracted = extract_text_from_pdf_bytes(data)
        if not extracted.strip():
            st.error("No selectable text found in PDF (probably scanned). For OCR you'd need Tesseract or another OCR pipeline.")
        else:
            st.success("Text extracted from PDF (preview):")
            st.text_area("Preview", extracted[:3000], height=200)
            if st.button("Create / Update Vector DB"):
                with st.spinner("Creating vector DB..."):
                    try:
                        vectordb = build_vectordb_from_text(extracted, persist_dir=CHROMA_DIR)
                        st.success(f"Chroma DB written to {CHROMA_DIR}")
                    except Exception as e:
                        st.error("Failed to build vectordb: " + str(e))
    else:
        txt = data.decode("utf-8")
        st.text_area("Preview", txt[:3000], height=200)
        if st.button("Create / Update Vector DB"):
            with st.spinner("Creating vector DB..."):
                try:
                    vectordb = build_vectordb_from_text(txt, persist_dir=CHROMA_DIR)
                    st.success(f"Chroma DB written to {CHROMA_DIR}")
                except Exception as e:
                    st.error("Failed to build vectordb: " + str(e))

st.markdown("---")
st.header("Questions (After DB exists)")
question = st.text_input("Ask a question about the uploaded document:")
if st.button("Ask") and question.strip():
    if not os.path.exists(CHROMA_DIR):
        st.error("Chroma DB not found. Upload file and create DB first.")
    else:
        with st.spinner("Running retrieval + Ollama LLM..."):
            try:
                embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
                vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
                qa = create_qa_chain_from_chroma(vectordb)
                answer = qa.run(question)
                st.subheader("Answer")
                st.write(answer)
            except Exception as e:
                st.error("Error: " + str(e))
