from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


def get_answer(vectorstore: FAISS, question: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Use only the context below to answer the question.
    If the answer is not in the context, say "I don't know based on this document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)