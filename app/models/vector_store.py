import os
import shutil
import chromadb
import chromadb.api
import uuid
import gc
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStoreManager:
    """
    Level 1 — Storage Layer.
    Responsible for all vector store lifecycle management:
    creating, persisting, and destroying Chroma sessions.
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db_root = "chroma_db_sessions"
        self.client = None
        self.session_path = None

        # Ensure the root sessions directory exists
        os.makedirs(self.db_root, exist_ok=True)

    def create_vector_store(self, docs) -> Chroma:
        """
        Creates a new isolated session folder for each uploaded document.
        Cleans up the previous session (memory + disk) before creating a new one.
        Returns a ready-to-use Chroma vectorstore instance.
        """
        # Always clean up previous session before creating a new one
        self._release_and_cleanup()

        # Unique session ID prevents WinError 32 file-lock conflicts
        session_id = str(uuid.uuid4())[:8]
        self.session_path = os.path.join(self.db_root, f"session_{session_id}")
        os.makedirs(self.session_path, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.session_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=self.client,
            collection_name="knowledge_base"
        )

        print(f"✅ Level 1: Vector store created at '{self.session_path}' "
              f"with {len(chunks)} chunks.")
        return vectorstore

    def _release_and_cleanup(self):
        """
        FIX #1 (duplicate delete_db) + FIX #5 (session folders never deleted):
        Single method that:
          1. Releases the Windows file lock on the sqlite3 DB
          2. Deletes the old session folder from disk entirely
        """
        # Step 1: Release internal ChromaDB cache (critical for Windows)
        chromadb.api.client.SharedSystemClient.clear_system_cache()

        if self.client:
            try:
                self.client._system.stop()
                print("🔓 Level 1: ChromaDB file locks released.")
            except Exception as e:
                print(f"⚠️  Cleanup warning (lock release): {e}")
            finally:
                self.client = None
                gc.collect()
                time.sleep(0.5)  # Buffer for Windows I/O to catch up

        # Step 2: Delete the old session folder from disk to prevent accumulation
        if self.session_path and os.path.exists(self.session_path):
            try:
                shutil.rmtree(self.session_path)
                print(f"🗑️  Level 1: Deleted old session folder '{self.session_path}'.")
            except Exception as e:
                print(f"⚠️  Cleanup warning (folder delete): {e}")
            finally:
                self.session_path = None

    def delete_db(self):
        """
        Public method called by main.py on /new_chat or before a fresh upload.
        Delegates to the internal cleanup method.
        """
        self._release_and_cleanup()
        print("🏁 Level 1: Session fully cleared. Ready for new document.")