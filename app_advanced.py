import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# PAGE CONFIG
st.set_page_config(page_title="🔥 AI PDF Chatbot", layout="wide")
st.markdown("<h1 style='color:#00FFD1;'>🤖 AI PDF Chatbot (Fast + AI)</h1>", unsafe_allow_html=True)


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings()


@st.cache_resource
def load_llm():
    return Ollama(model="phi")


@st.cache_resource
def process_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    embeddings = load_embeddings()
    db = FAISS.from_texts(chunks, embeddings)
    return db


uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")

if uploaded_file:
    if "db" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        st.info("⏳ Processing PDF... please wait")
        st.session_state.db = process_pdf(uploaded_file)
        st.session_state.last_file = uploaded_file.name

    st.success("✅ PDF Ready! Ask your question below")

    llm = load_llm()
    db = st.session_state.db

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("💬 Ask something about your PDF...")

    if query:
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.write(query)

        docs = db.similarity_search(query, k=2)
        context = " ".join([doc.page_content[:1000] for doc in docs])

        prompt = f"""
        Answer clearly using only the context below.

        Context:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        with st.chat_message("assistant"):
            st.write(response)
