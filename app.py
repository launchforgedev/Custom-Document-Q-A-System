import os
import shutil
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="Custom Document QA System", page_icon="📄")
st.title("📄 Custom Document QA System")
st.write("Upload a PDF document and search for relevant context using vector similarity.")

# ChromaDB persistence directory
DB_DIR = "./chroma_db"

# Load HuggingFace embeddings once globally using Streamlit caching
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Sidebar for file upload
st.sidebar.header("Document Upload")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

def process_pdf(pdf_file):
    # 1. Clean up existing ChromaDB to prevent duplicate data accumulation
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    # 2. Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        # 3. Create fresh vector database
        vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
        return vector_db
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if uploaded_file is not None:
    with st.spinner("Processing PDF and generating vector embeddings..."):
        vector_db = process_pdf(uploaded_file)
    st.sidebar.success("PDF processed and indexed successfully!")

    # Search Query Interface
    query = st.text_input("Ask a question about your document:")
    
    if st.button("Search"):
        if query.strip() != "":
            # Search for top matches
            raw_docs = vector_db.similarity_search(query, k=5)
            
            # Deduplicate results based on exact text content
            unique_contexts = []
            seen_content = set()
            for doc in raw_docs:
                clean_text = doc.page_content.strip()
                if clean_text not in seen_content:
                    seen_content.add(clean_text)
                    unique_contexts.append(clean_text)
            
            st.markdown("### 🔍 Relevant Context Retrieved:")
            if unique_contexts:
                # Display only 1 single unique chunk
                st.markdown(unique_contexts[0])
            else:
                st.info("No matching context found.")
        else:
            st.warning("Please enter a question to search.")
else:
    st.info("Please upload a PDF file from the sidebar to begin.")