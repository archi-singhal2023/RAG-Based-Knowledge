# DocuMind 🧠

An open-source, locally deployable AI system that lets you upload a PDF and chat with it using natural language.

🚀 **Live Demo:** [DocuMind on HuggingFace Spaces](https://huggingface.co/spaces/your-username/documind)

---

## What it does

Upload any PDF research paper and ask questions about it in plain English. DocuMind retrieves relevant sections and generates accurate, context-aware answers grounded exclusively in the document content.

---

## Features

- 📄 Upload any PDF and start chatting instantly
- 🔍 Semantic search using vector embeddings
- 🧠 Powered by Mistral-7B via HuggingFace
- 🔒 Fully local — your documents never leave your machine
- 💬 Conversation memory for context-aware follow-up questions
- 🚫 Refuses to hallucinate — says "I don't know" when information is not in the document
- 🗂️ Session isolation — each document gets its own isolated vector store
- ☁️ Optional AWS S3 backup

---

## Tech Stack

| Component       | Technology                     |
| --------------- | ------------------------------ |
| Backend         | Python, Flask                  |
| Vector Database | ChromaDB                       |
| Embeddings      | paraphrase-MiniLM-L3-v2        |
| Language Model  | Mistral-7B-Instruct-v0.2       |
| Frontend        | HTML5, Bootstrap 5, JavaScript |
| Deployment      | Docker, HuggingFace Spaces     |

---

## Project Structure

```
RAG-Based-Knowledge/
├── app/
│   ├── models/
│   │   └── vector_store.py
│   ├── service/
│   │   ├── llm_service.py
│   │   └── storage_service.py
│   ├── static/
│   │   ├── style.css
│   │   └── app.js
│   ├── templates/
│   │   └── index.html
│   └── config.py
├── main.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Running Locally

### Prerequisites

- Python 3.11+
- HuggingFace API token

### Setup

```bash
# Clone the repository
git clone https://github.com/archi-singhal2023/RAG-Based-Knowledge.git
cd RAG-Based-Knowledge

# Create virtual environment
conda create -n rag_env python=3.11
conda activate rag_env

# Install dependencies
pip install -r requirements.txt

# Add environment variables
cp .env.example .env
# Edit .env and add your HUGGINGFACEHUB_API_TOKEN
```

### Run

```bash
python main.py
```

Open `http://localhost:8080` in your browser.

### Run with Docker

```bash
docker build -t documind .
docker run -p 7860:7860 --env-file .env documind
```
