# app.py
import streamlit as st
from utils.qa_chain import get_chain

st.set_page_config(page_title="Loan Policy Assistant", layout="centered")
st.title("💸 Loan Policy Assistant")

qa_chain = get_chain()

query = st.text_input("Ask me anything about loan policies:")

if query:
    with st.spinner("Finding the best answer..."):
        answer = qa_chain.run(query)
        st.success(answer)
