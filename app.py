import streamlit as st
import faiss
import numpy as np
from mistralai import Mistral, UserMessage
from bs4 import BeautifulSoup
import requests

API_KEY = "WDtXdrmUGvwuVnXxTAhlZUZmMbXdyopZ"

def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=API_KEY)
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return embeddings_batch_response.data

def mistral_chat(user_message):
    client = Mistral(api_key=API_KEY)
    messages = [UserMessage(content=user_message)]
    chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
    return chat_response.choices[0].message.content

def load_policy_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    return text

def build_faiss_index(text_chunks):
    text_embeddings = get_text_embedding(text_chunks)
    embeddings = np.array([emb.embedding for emb in text_embeddings])
    index = faiss.IndexFlatL2(len(text_embeddings[0].embedding))
    index.add(embeddings)
    return index, text_chunks

def retrieve_relevant_chunks(query, index, text_chunks, top_k=2):
    query_embedding = np.array([get_text_embedding([query])[0].embedding])
    D, I = index.search(query_embedding, k=top_k)
    return [text_chunks[i] for i in I.tolist()[0]]

# Streamlit UI
st.title("Policy ChatBot Powered by UDST")

# Policy selection
policy_urls = {
    "1- Student Conduct Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy",
    "2- Academic Schedule Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy",
    "3- Student Attendance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
    "4- Student Appeals Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "5- Graduation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
    "6- Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-standing-policy",
    "7- Transfer Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy",
    "8- Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/admissions-policy",
    "9- Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
    "10- Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
}

selected_policy = st.selectbox("Policy Selection:", list(policy_urls.keys()))

if st.button("Load"):
    policy_text = load_policy_data(policy_urls[selected_policy])
    chunks = [policy_text[i:i+512] for i in range(0, len(policy_text), 512)]
    index, chunks = build_faiss_index(chunks)
    st.session_state["index"] = index
    st.session_state["chunks"] = chunks
    st.success("Policy data loaded successfully!")

# Query input
query = st.text_input("Ask a Question:")
if st.button("Get Answer") and "index" in st.session_state:
    retrieved_chunks = retrieve_relevant_chunks(query, st.session_state["index"], st.session_state["chunks"])
    prompt = f"""
    Context information:
    {retrieved_chunks}
    Given the context, answer the query: {query}
    """
    response = mistral_chat(prompt)
    st.text_area("Answer:", response, height=200)
