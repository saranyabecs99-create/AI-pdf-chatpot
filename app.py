import streamlit as st
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# =========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

# =========================================================
# 2. STREAMLIT PAGE
# =========================================================

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI PDF Chatbot")
st.write("Upload a PDF and ask questions from the document.")

# =========================================================
# 3. PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

# =========================================================
# 4. PROCESS PDF
# =========================================================

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    pdf_path = "uploaded_file.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully! ✅")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    st.info(f"PDF processed into {len(chunks)} text chunks.")

    # =====================================================
    # 5. EMBEDDINGS
    # =====================================================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # =====================================================
    # 6. VECTOR DATABASE
    # =====================================================

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_chatbot"
    )

    # =====================================================
    # 7. OLLAMA MODEL
    # =====================================================

    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    # =====================================================
    # 8. QUESTION INPUT
    # =====================================================

    question = st.text_input(
        "Ask a question about your PDF:"
    )

    if question:

        with st.spinner("Thinking... 🤖"):

            # Search relevant documents
            results = vector_db.similarity_search(
                question,
                k=4
            )

            # Combine retrieved text
            context = "\n\n".join(
                [doc.page_content for doc in results]
            )

            # =================================================
            # 9. PROMPT
            # =================================================

            prompt = f"""
You are an AI assistant that answers questions based only
on the uploaded PDF.

Context from PDF:
{context}

Question:
{question}

Instructions:
- Answer clearly and simply.
- Use only the information from the PDF.
- If the answer is not available in the PDF, say:
  "The answer is not available in the uploaded PDF."
"""

            # =================================================
            # 10. GET ANSWER
            # =================================================

            response = llm.invoke(prompt)

            st.subheader("🤖 Answer")

            st.write(response.content)

else:

    st.info("Please upload a PDF to start chatting. 📄")
    
