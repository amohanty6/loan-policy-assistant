# utils/qa_chain.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub  # Use any HF model

def get_chain():
    db = FAISS.load_local("vector_store", HuggingFaceEmbeddings())
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.3, "max_length": 256}),
        retriever=retriever
    )
    return qa
