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
    # Save uploaded file temporarily to extract text
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        # Create an in-memory Chroma instance (No persistent directory to corrupt/lock)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        return vector_db
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if uploaded_file is not None:
    # Process PDF only if it hasn't been stored in session_state yet or if a new file is uploaded
    if "vector_db" not in st.session_state or st.session_state.get("last_uploaded_file") != uploaded_file.name:
        with st.spinner("Processing PDF and generating vector embeddings..."):
            st.session_state.vector_db = process_pdf(uploaded_file)
            st.session_state.last_uploaded_file = uploaded_file.name
        st.sidebar.success("PDF processed and indexed successfully!")

    # Search Query Interface
    query = st.text_input("Ask a question about your document:")
    
    if st.button("Search"):
        if query.strip() != "":
            # Retrieve search results from in-memory Chroma instance stored in session_state
            vector_db = st.session_state.vector_db
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
                # Display 1 single unique chunk output
                st.markdown(unique_contexts[0])
            else:
                st.info("No matching context found.")
        else:
            st.warning("Please enter a question to search.")
else:
    st.info("Please upload a PDF file from the sidebar to begin.")