# 📄 Custom PDF Query & Retrieval System (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that allows users to upload custom PDF documents and perform semantic vector search to retrieve precise context.

## 🚀 Key Features
* **Document Chunking:** Recursive text splitting with dynamic overlap.
* **Vector Embeddings:** Open-source `all-MiniLM-L6-v2` embeddings via Hugging Face.
* **Vector Store:** Local persistent storage using ChromaDB.
* **UI Interface:** Interactive user portal built with Gradio.

## 🛠 Tech Stack
* **Language:** Python 3.10+
* **Frameworks:** LangChain, Hugging Face Transformers
* **Database:** ChromaDB
* **UI:** Gradio

## 🚦 Quickstart

```bash
# Clone the repository
git clone [https://github.com/your-username/rag-document-qa.git](https://github.com/your-username/rag-document-qa.git)
cd rag-document-qa

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py