import os
import shutil
import chromadb
import gc
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db_path = "chroma_db"
        self.client = None

    def load_vector_store(self):
        """Safe connection for server startup."""
        if os.path.exists(self.db_path) and os.listdir(self.db_path):
            try:
                self.client = chromadb.PersistentClient(path=self.db_path)
                return Chroma(
                    client=self.client,
                    collection_name="document_collection",
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"⚠️ Vector store corrupted: {e}")
                return None
        return None

    def create_vector_store(self, docs):
        # 1. ALWAYS wipe the old DB before creating a new one to prevent 'Context Soup'
        self.delete_db()
        
        # 2. Re-initialize the persistent client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, 
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=self.client, 
            collection_name="document_collection"
        )
        print(f"✅ Indexed {len(chunks)} chunks.")
        return vectorstore

    def delete_db(self):
        """Explicitly stops Chroma and wipes the folder to release Windows locks."""
        if self.client:
            try:
                self.client._system.stop() # Releases .bin and .sqlite3 files 
                self.client = None
                gc.collect() # Force-release file handles 
                print("🏁 File locks released.")
            except Exception as e:
                print(f"Cleanup warning: {e}")

        # Physically remove the folder so the next upload is 100% fresh
        if os.path.exists(self.db_path):
            try:
                shutil.rmtree(self.db_path)
                print("🧹 Database folder deleted.")
            except Exception as e:
                print(f"⚠️ Manual wipe failed (file still locked): {e}")