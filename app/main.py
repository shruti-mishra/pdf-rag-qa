import streamlit as st
import tempfile
import os
from ingest import ingest_pdf
from rag_chain import get_answer

# Page config
st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 PDF Q&A — RAG Demo")
st.caption("Upload a PDF and ask questions about its content")

# Upload widget
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Save uploaded file to a temp location so PyMuPDF can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Only re-process if a new file is uploaded
    if "vectorstore" not in st.session_state or \
       st.session_state.get("filename") != uploaded_file.name:
        with st.spinner("Reading and indexing your PDF..."):
            st.session_state.vectorstore = ingest_pdf(tmp_path)
            st.session_state.filename = uploaded_file.name
        st.success("PDF indexed! Ask your question below.")

    os.unlink(tmp_path)  # Clean up the temp file

    # Question input
    question = st.text_input("Ask a question about the document")

    if question:
        with st.spinner("Thinking..."):
            answer = get_answer(st.session_state.vectorstore, question)
        st.markdown("### Answer")
        st.write(answer)

else:
    st.info("Upload a PDF above to get started.")