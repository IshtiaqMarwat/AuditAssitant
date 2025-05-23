# -*- coding: utf-8 -*-
"""Audit Assistant Chatbot"""

# pip install -U streamlit langchain langchain-openai langchain-community PyPDF2 faiss-cpu sentence-transformers

import streamlit as st
import PyPDF2
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 🔐 Load API Key ---
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ OpenAI API key not found. Please set it in Streamlit secrets.")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# --- 🤖 Initialize LLM ---
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # or "gpt-4o"
    temperature=0,
    api_key=openai_api_key
)

# --- 📄 PDF Text Extractor ---
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# --- 🧠 Embeddings + Splitter ---
embeddings = HuggingFaceEmbeddings()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len
)

def ask_audit_question(pdf_text, question):
    chunks = text_splitter.split_text(pdf_text)
    db = FAISS.from_texts(chunks, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain.invoke(question)["result"]

# --- 🌐 Streamlit UI ---
st.title("🕵️ AI Audit Assistant")
st.markdown("Upload an audit policy PDF and ask questions about it.")

uploaded_file = st.file_uploader("📎 Upload your Audit PDF", type="pdf")

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ Document uploaded and processed!")

    user_question = st.text_input("💬 Ask a question about the document:")

    if user_question:
        with st.spinner("Thinking..."):
            try:
                answer = ask_audit_question(pdf_text, user_question)
                st.text_area("📘 Audit Assistant's Answer", value=answer, height=200)
            except Exception as e:
                st.error(f"❌ Error: {e}")
