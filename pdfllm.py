#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import os

# Set Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# LLM
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 300}
)

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(pdf):
    doc = fitz.open(stream=pdf.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to build RAG answer
def generate_answer(query, text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embedding=embedding_model)
    docs = vector_store.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:
"""
    response = llm(prompt)
    return response.split("Answer:")[-1].strip()

# Streamlit UI
st.title("📄 Chat with Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("Ask a question about the document:")

if uploaded_file and query:
    with st.spinner("Thinking..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        answer = generate_answer(query, pdf_text)
        st.success("Answer:")
        st.write(answer)

