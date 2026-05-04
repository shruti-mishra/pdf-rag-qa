# 📄 PDF Q&A — Retrieval-Augmented Generation (RAG) Demo

> Upload any PDF and ask questions about its content using AI. Built as part of my transition into AI/ML engineering.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-000000?style=flat)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3-F55036?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-0467DF?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat&logo=streamlit&logoColor=white)

---

## What it does

This app lets you upload any PDF — a research paper, a contract, a textbook chapter — and ask plain English questions about its content. The answers are grounded strictly in the document, not in the model's general knowledge.

**Example use cases:**
- "What are the key findings of this research paper?"
- "What does clause 4.2 of this contract say about termination?"
- "Summarise the methodology section of this report"

---

## How it works — RAG pipeline

```
PDF upload → Text extraction → Chunking → Embeddings → FAISS vector store
                                                               ↓
User question → Query embedding → Similarity search → Top-4 chunks
                                                               ↓
                                          Groq LLaMA 3.3 → Answer
```

1. **Ingestion** — PyMuPDF extracts raw text from the uploaded PDF
2. **Chunking** — LangChain splits the text into 500-character overlapping chunks
3. **Embedding** — HuggingFace `all-MiniLM-L6-v2` converts chunks into vectors (runs locally, no API cost)
4. **Vector store** — FAISS indexes all vectors for fast similarity search
5. **Retrieval** — On each question, the 4 most relevant chunks are retrieved
6. **Generation** — Groq's LLaMA 3.3 70B model generates an answer grounded in those chunks

---

## Tech stack

| Layer | Tool | Purpose |
|---|---|---|
| UI | Streamlit | Web interface |
| PDF parsing | PyMuPDF (fitz) | Text extraction |
| Text splitting | LangChain | Chunking with overlap |
| Embeddings | HuggingFace sentence-transformers | Local vector generation |
| Vector store | FAISS | Similarity search |
| LLM | Groq — LLaMA 3.3 70B | Answer generation |
| Orchestration | LangChain LCEL | Chain composition |

---

## Project structure

```
pdf-rag-qa/
├── app/
│   ├── main.py           # Streamlit UI
│   ├── ingest.py         # PDF → chunks → FAISS vector store
│   └── rag_chain.py      # Query → retrieval → LLM answer
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Run it locally

**1. Clone the repo**
```bash
git clone git@github.com:shruti-mishra/pdf-rag-qa.git
cd pdf-rag-qa
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/Scripts/activate  # Windows (Git Bash)
# or
source venv/bin/activate       # Mac / Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your API key**
```bash
cp .env.example .env
# Open .env and add your Groq API key
# Get a free key at https://console.groq.com
```

**5. Run the app**
```bash
cd app
streamlit run main.py
```

Open `http://localhost:8501` in your browser.

---

## Configuration

Create a `.env` file in the project root:

```
GROQ_API_KEY=your-groq-api-key-here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

---

## What I learned building this

This project is part of my hands-on transition into AI/ML engineering. Key things I worked through:

- **RAG architecture** — understanding why retrieval matters and how chunking strategy affects answer quality
- **Vector embeddings** — how text gets converted into numerical representations and why similar meanings end up close together in vector space
- **LangChain LCEL** — composing chains using the modern pipe syntax instead of deprecated wrapper classes
- **LLM prompt design** — constraining the model to answer only from retrieved context, reducing hallucination
- **Debugging dependency conflicts** — LangChain's fast-moving ecosystem meant navigating several breaking import changes

---

## Limitations and future improvements

- [ ] Vector store is in-memory only — resets on each upload. Next step: persist with ChromaDB
- [ ] No conversation memory — each question is independent. Next step: add chat history with LangChain memory
- [ ] Single PDF only. Next step: support multiple documents
- [ ] Add Docker support for one-command deployment
- [ ] Add GitHub Actions CI to run tests on every push

---

## Author

**Shruti Mishra** — transitioning into AI/ML engineering, building projects in LLMs, RAG, and applied NLP.

Connect on [LinkedIn](www.linkedin.com/in/shruti-mishra-82b271107) · [GitHub](https://github.com/shruti-mishra)

---

*Free to use for learning and portfolio purposes.*