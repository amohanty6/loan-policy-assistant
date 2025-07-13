import streamlit as st
import os
from utils.qa_chain import get_chain

st.set_page_config(page_title="Loan Policy Assistant", layout="centered")
st.title("💸 Loan Policy Assistant")

# 🔁 Optional: Trigger ingestion
if st.button("⚙️ Ingest Documents"):
    with st.spinner("Ingesting documents and building vector store..."):
        exit_code = os.system("python ingest.py")
        if exit_code == 0:
            st.success("✅ Documents ingested successfully!")
        else:
            st.error("⚠️ Failed to ingest documents. Check logs or file paths.")

# ✅ Load QA chain
try:
    qa_chain = get_chain()
except Exception as e:
    st.error("❌ Error loading QA chain. Have you ingested the documents?")
    st.stop()

# 🧠 Ask a question
query = st.text_input("Ask me anything about loan policies:")

if query:
    with st.spinner("Searching documents and generating answer..."):
        try:
            answer = qa_chain.run(query)
            st.success(answer)
        except Exception as e:
            st.error(f"❌ Failed to answer: {e}")
