import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_and_chunk_pdf(file_path: str) -> list:
    # Open the PDF and extract all text page by page
    doc = fitz.open(file_path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    # Split text into overlapping chunks so context isn't lost at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # ~500 characters per chunk
        chunk_overlap=50     # 50 characters overlap between chunks
    )
    chunks = splitter.create_documents([full_text])
    return chunks


def build_vectorstore(chunks: list) -> FAISS:
    # Use a free HuggingFace model to convert chunks into vectors
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Store all vectors in FAISS (runs locally, no API needed)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def ingest_pdf(file_path: str) -> FAISS:
    chunks = load_and_chunk_pdf(file_path)
    vectorstore = build_vectorstore(chunks)
    return vectorstore