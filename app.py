import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import gradio as gr

# 1. Load Document
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # 2. Chunk Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    # 3. Store in Vector Database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    return vector_db

# 4. Search Function
def ask_question(vector_db, query):
    docs = vector_db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

def qa_pipeline(pdf_file, question):
    vector_db = process_pdf(pdf_file.name)
    relevant_context = ask_question(vector_db, question)
    return f"**Relevant Context Retrieved:**\n\n{relevant_context}"

demo = gr.Interface(
    fn=qa_pipeline,
    inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Ask a Question")],
    outputs="markdown",
    title="Custom Document QA System"
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
