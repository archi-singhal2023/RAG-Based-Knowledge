import os
import chromadb
import gc, time
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db_root = "chroma_db_sessions"
        self.current_session_path = None
        self.session_path = None
        self.client = None
        

    def create_vector_store(self, docs):
        # 1. Force release existing locks before starting new session
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        if self.client:
            self.client._system.stop()
            self.client = None
            gc.collect()
            time.sleep(1)
        
        # 1. Level 1 Isolation: Generate a unique ID for this specific document
        session_id = str(uuid.uuid4())[:8]
        session_path = os.path.join(self.db_root, f"session_{session_id}")
        
        # 2. Initialize the persistent client
        self.client = chromadb.PersistentClient(path=session_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=self.client, 
            collection_name="knowledge_base"
        )
    def load_vector_store(self):
        """Level 1: Reconnects to an existing session on startup or new chat."""
        target_path = self.session_path or self.current_session_path
        if target_path and os.path.exists(target_path):
            try:
                self.client = chromadb.PersistentClient(path=target_path)
                return Chroma(
                    client=self.client,
                    collection_name="knowledge_base",
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"⚠️ Level 1 Error: Could not load session: {e}")
                return None
        return None
    
    def delete_db(self):
        """Level 3: Clears memory and internal caches to release Windows locks."""
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        if self.client:
            try:
                # Explicitly stop the system to release the .sqlite3 file
                self.client._system.stop()
                self.client = None
                gc.collect() 
                time.sleep(1) # Essential buffer for Windows I/O
                print("🏁 Level 3: Windows file locks released.")
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")
    