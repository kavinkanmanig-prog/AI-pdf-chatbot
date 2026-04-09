import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# 🎨 PAGE CONFIG
st.set_page_config(page_title="🔥 AI PDF Chatbot", layout="wide")

# 🌙 SIMPLE PREMIUM TITLE
st.markdown("<h1 style='color:#00FFD1;'>🤖 AI PDF Chatbot (Fast + AI)</h1>", unsafe_allow_html=True)

# ✅ CACHE PDF PROCESSING (VERY IMPORTANT)
@st.cache_resource
def process_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    # ✂️ Split text
    splitter = CharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )
    chunks = splitter.split_text(text)

    # 🧠 Embeddings + DB
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(chunks, embeddings)

    return db

# 📄 FILE UPLOAD
uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")

if uploaded_file:
    st.info("⏳ Processing PDF... please wait")

    db = process_pdf(uploaded_file)

    st.success("✅ PDF Ready! Ask your question below")

    # ⚡ FAST MODEL (STEP 2 INCLUDED)
    llm = Ollama(model="phi")

    # 💬 CHAT MEMORY
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # SHOW CHAT HISTORY
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # USER INPUT
    query = st.chat_input("💬 Ask something about your PDF...")

    if query:
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.write(query)

        # 🔍 RETRIEVE CONTEXT (STEP 3 OPTIMIZED)
        docs = db.similarity_search(query, k=2)
        context = " ".join([doc.page_content for doc in docs])

        # 🧠 PROMPT
        prompt = f"""
        Answer clearly using only the context below.

        Context:
        {context}

        Question:
        {query}
        """

        # 🤖 AI RESPONSE
        response = llm.invoke(prompt)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        with st.chat_message("assistant"):
            st.write(response)