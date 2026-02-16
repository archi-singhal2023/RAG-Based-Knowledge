import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorStoreManager:
    def __init__(self):
        # Using a high-quality, free embedding model from Hugging Face
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db_path = "chroma_db" # This will be created locally

    def create_vector_store(self, docs):
        """
        Takes list of documents, splits them, and stores them in ChromaDB
        """
        # 1. Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        
        # 2. Create and persist the vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print(f"✅ Vector store created with {len(chunks)} chunks.")
        return vectorstore

    def load_vector_store(self):
        """
        Loads the existing vector store from disk
        """
        if os.path.exists(self.db_path):
            return Chroma(
                persist_directory=self.db_path, 
                embedding_function=self.embeddings
            )
        return None