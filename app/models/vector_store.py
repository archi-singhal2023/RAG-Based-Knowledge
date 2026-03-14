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

    Windows SQLite lock strategy:
    Rather than fighting the lock immediately after closing ChromaDB,
    we defer deletion of the old session folder to a background thread
    that retries with generous backoff. The app never blocks on cleanup.
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db_root = "chroma_db_sessions"
        self.client = None
        self.session_path = None
        os.makedirs(self.db_root, exist_ok=True)

        # On startup, clean up any orphaned session folders from previous runs
        self._cleanup_orphaned_sessions()

    def create_vector_store(self, docs) -> Chroma:
        """
        Creates a new isolated session. Releases the previous client lock
        synchronously, then delegates folder deletion to a background thread
        so the upload response is never blocked by Windows I/O.
        """
        old_path = self._release_client()  # Synchronous lock release only

        if old_path:
            # Kick off folder deletion in background — no blocking, no warnings
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
        """
        Synchronously stops the ChromaDB client and clears its internal cache.
        Returns the old session path so the caller can schedule deletion.
        Does NOT delete the folder — that's done separately by background thread.
        """
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
        """
        Background thread target. Retries folder deletion with exponential
        backoff for up to ~30 seconds total. SQLite releases its lock within
        1-3 seconds on Windows after the client is stopped — this comfortably
        covers that window without blocking anything.
        """
        # Initial wait: give Windows time to release the SQLite lock
        time.sleep(2)

        max_retries = 8
        for attempt in range(1, max_retries + 1):
            if not os.path.exists(path):
                return  # Already gone — nothing to do

            try:
                shutil.rmtree(path)
                print(f"🗑️  Level 1: Deleted session folder '{os.path.basename(path)}' "
                      f"(attempt {attempt}).")
                return  # Success
            except PermissionError:
                if attempt == max_retries:
                    print(f"⚠️  Level 1: Could not delete '{path}' after {max_retries} "
                          f"attempts. It will be cleaned up on next app start.")
                else:
                    wait = min(2 ** attempt, 10)  # 2s, 4s, 8s, 10s, 10s...
                    print(f"⏳ Folder locked, retry {attempt}/{max_retries} in {wait}s...")
                    time.sleep(wait)
            except Exception as e:
                print(f"⚠️  Unexpected folder delete error: {e}")
                return

    def _cleanup_orphaned_sessions(self):
        """
        Called once on startup. Deletes any leftover session folders from
        previous runs where the app was killed before cleanup could finish.
        These folders have no active lock so deletion always succeeds.
        """
        if not os.path.exists(self.db_root):
            return
        for name in os.listdir(self.db_root):
            if name.startswith("session_"):
                orphan = os.path.join(self.db_root, name)
                try:
                    shutil.rmtree(orphan)
                    print(f"🧹 Level 1: Cleaned up orphaned session '{name}' from previous run.")
                except Exception as e:
                    print(f"⚠️  Could not clean orphan '{name}': {e}")

    def delete_db(self):
        """Public method called by main.py on /new_chat."""
        old_path = self._release_client()
        if old_path:
            t = threading.Thread(
                target=self._delete_folder_with_retry,
                args=(old_path,),
                daemon=True
            )
            t.start()
        print("🏁 Level 1: Session cleared. Folder deletion running in background.")