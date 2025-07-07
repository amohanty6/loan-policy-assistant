# ingest.py
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

DATA_FOLDER = "data/"
INDEX_FOLDER = "vector_store/"

def load_documents():
    documents = []
    for file in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, file)
        if file.endswith(".pdf"):
            documents += PyPDFLoader(path).load()
        elif file.endswith(".txt"):
            documents += TextLoader(path).load()
        # Add more loaders as needed
    return documents

def ingest():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_FOLDER)

if __name__ == "__main__":
    ingest()
