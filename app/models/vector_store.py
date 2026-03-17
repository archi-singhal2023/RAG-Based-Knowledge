import os
import shutil
import chromadb
import chromadb.api
import uuid
import gc
import time
import threading
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStoreManager:
    """
    Level 1 — Storage Layer.
    Responsible for all vector store lifecycle: creating, persisting, destroying.
    """

    def __init__(self):
        self._embeddings = None
        self.db_root = os.getenv("CHROMA_DB_PATH", "/data/chroma_db_sessions")
        self.client = None
        self.session_path = None
        os.makedirs(self.db_root, exist_ok=True)
        self._cleanup_orphaned_sessions()

    @property
    def embeddings(self):
        if self._embeddings is None:
            print("📦 Loading embedding model...")
            # Use /app/models if it exists (inside Docker)
            # Otherwise let sentence-transformers use its default cache
            cache_folder = "/app/models" if os.path.exists("/app/models") else None
            self._embeddings = HuggingFaceEmbeddings(
                model_name="paraphrase-MiniLM-L3-v2",
                cache_folder=cache_folder
            )
            print("✅ Embedding model loaded.")
        return self._embeddings

    def create_vector_store(self, docs) -> Chroma:
        old_path = self._release_client()

        if old_path:
            t = threading.Thread(
                target=self._delete_folder_with_retry,
                args=(old_path,),
                daemon=True
            )
            t.start()

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

    def _release_client(self) -> str | None:
        chromadb.api.client.SharedSystemClient.clear_system_cache()

        old_path = None
        if self.client:
            try:
                self.client._system.stop()
                print("🔓 Level 1: ChromaDB client stopped.")
            except Exception as e:
                print(f"⚠️  Client stop warning: {e}")
            finally:
                old_path = self.session_path
                self.client = None
                self.session_path = None
                gc.collect()

        return old_path

    def _delete_folder_with_retry(self, path: str):
        time.sleep(2)
        max_retries = 8
        for attempt in range(1, max_retries + 1):
            if not os.path.exists(path):
                return
            try:
                shutil.rmtree(path)
                print(f"🗑️  Level 1: Deleted session folder '{os.path.basename(path)}' "
                      f"(attempt {attempt}).")
                return
            except PermissionError:
                if attempt == max_retries:
                    print(f"⚠️  Could not delete '{path}' after {max_retries} attempts.")
                else:
                    wait = min(2 ** attempt, 10)
                    time.sleep(wait)
            except Exception as e:
                print(f"⚠️  Unexpected folder delete error: {e}")
                return

    def _cleanup_orphaned_sessions(self):
        if not os.path.exists(self.db_root):
            return
        for name in os.listdir(self.db_root):
            if name.startswith("session_"):
                orphan = os.path.join(self.db_root, name)
                try:
                    shutil.rmtree(orphan)
                    print(f"🧹 Level 1: Cleaned up orphaned session '{name}'.")
                except Exception as e:
                    print(f"⚠️  Could not clean orphan '{name}': {e}")

    def delete_db(self):
        old_path = self._release_client()
        if old_path:
            t = threading.Thread(
                target=self._delete_folder_with_retry,
                args=(old_path,),
                daemon=True
            )
            t.start()
        print("🏁 Level 1: Session cleared.")