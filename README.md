# AI-Powered PDF Question Answering System

An intelligent PDF chatbot built using Retrieval-Augmented Generation (RAG) and a Local Large Language Model (LLM). This application allows users to upload PDF documents and ask natural language questions about their contents through an interactive chatbot interface.

## Features

- Upload PDF documents
- Extract and process PDF text
- Semantic chunking and embedding generation
- FAISS vector database for similarity search
- Local LLM inference using Ollama
- ChatGPT-like Streamlit interface
- Offline / privacy-preserving execution

## Tech Stack

- Python
- Streamlit
- Ollama
- Phi / LLaMA 3
- LangChain
- FAISS
- Sentence Transformers
- pypdf

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd AI_PDF_CHATBOT_ADVANCED
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama and pull the model:
```bash
ollama run phi
```

4. Run the application:
```bash
python -m streamlit run app_advanced.py
```

## Usage

1. Launch the application
2. Upload a PDF file
3. Wait for processing
4. Ask questions related to the document
5. Receive AI-generated answers

## Project Workflow

1. PDF Upload
2. Text Extraction
3. Text Chunking
4. Embedding Generation
5. Vector Storage (FAISS)
6. Semantic Retrieval
7. Response Generation via Local LLM

## Team Members

- Kavin Kanmani G
- Harish Ragavendran S

## License

This project is developed for academic and educational purposes.
