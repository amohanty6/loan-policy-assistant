import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_FOLDER = "data"
INDEX_FOLDER = "vector_store"

def load_documents():
    documents = []
    for file in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, file)
        if file.endswith(".pdf"):
            documents.extend(PyPDFLoader(file_path).load())
        elif file.endswith(".txt"):
            documents.extend(TextLoader(file_path).load())
    return documents

def ingest():
    docs = load_documents()
    if not docs:
        raise ValueError("No documents found to ingest!")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_FOLDER)

if __name__ == "__main__":
    ingest()
