import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="Custom Document QA System", page_icon="📄")
st.title("📄 Custom Document QA System")
st.write("Upload a PDF document and search for relevant context using vector similarity.")

# Load HuggingFace embeddings once globally using Streamlit caching
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Sidebar for file upload
st.sidebar.header("Document Upload")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

def process_pdf(pdf_file):
    # Save uploaded file temporarily to process with PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        vector_db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
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
            docs = vector_db.similarity_search(query, k=3)
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            
            st.markdown("### 🔍 Relevant Context Retrieved:")
            st.markdown(context)
        else:
            st.warning("Please enter a question to search.")
else:
    st.info("Please upload a PDF file from the sidebar to begin.")