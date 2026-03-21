import os
import gc
import logging
import tempfile
import threading
from flask import Flask, render_template, request, jsonify
from app.config import Config
from app.service.llm_service import LLMService
from app.models.vector_store import VectorStoreManager
from app.service.storage_service import StorageService
from langchain_community.document_loaders import PyPDFLoader
import uuid

_upload_jobs = {}
# Configure server-side logging (errors logged here, not exposed to client)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=os.path.join('app', 'templates'))

# ─── Application State ────────────────────────────────────────────────────────
# FIX #3 (thread safety): A lock ensures that concurrent /upload requests
# cannot corrupt shared state by interleaving their writes.
_state_lock = threading.Lock()

# Level 1 — Storage layer (manages vector store sessions)
vector_manager = VectorStoreManager()

# Level 2 — Intelligence layer (initialized once a document is uploaded)
# FIX #9: llm_service is reused across uploads; only its vectorstore is swapped.
llm_service: LLMService | None = None

def _upload_to_s3_background(temp_path: str, filename: str):
    """
    Silently uploads the PDF to S3 in a background thread.
    If AWS credentials are not configured, it logs and skips gracefully.
    The user never waits for this — it runs after the response is returned.
    """
    try:
        # Only attempt if AWS is fully configured
        if not all([Config.AWS_ACCESS_KEY_ID, Config.AWS_SECRET_ACCESS_KEY,
                    Config.BUCKET_NAME]):
            logger.info("S3 not configured — skipping cloud backup.")
            return

        storage = StorageService()

        # Save temp content to a new temp file since original may be deleted
        # by the time this thread runs
        import tempfile, shutil
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            backup_path = tmp.name

        shutil.copy2(temp_path, backup_path) if os.path.exists(temp_path) else None

        if os.path.exists(backup_path):
            success = storage.upload_pdf(backup_path, object_name=filename)
            if success:
                logger.info(f"S3 backup complete: '{filename}'")
            try:
                os.remove(backup_path)
            except:
                pass

    except Exception as e:
        # Never crash the app over a backup failure
        logger.warning(f"S3 background upload failed (non-critical): {e}")
# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global llm_service

    # FIX #7: Server-side file presence and type validation
    if 'file' not in request.files:
        return jsonify({"error": "No file was attached to the request."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported. Please upload a .pdf file."}), 400

    # FIX #4: Use NamedTemporaryFile to get a guaranteed unique temp path,
    # eliminating collisions from two uploads with the same filename.
    tmp_file = None
    try:
        with _state_lock:
            # Create a unique temp file
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_path = tmp_file.name
            tmp_file.close()  # Close before saving so Windows can write to it

            file.save(temp_path)

            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            if not docs:
                return jsonify({"error": "Could not extract text from this PDF. It may be scanned or image-based."}), 400

            # FIX (page-1 pinning): Extract Page 1 text directly from the loader.
            # PyPDFLoader produces one Document per page, so docs[0] is always
            # Page 1. We take pages 0-1 in case a long title spills onto Page 2.
            # This raw text is passed directly into LLMService, bypassing any
            # metadata filter that Chroma may or may not support.
            page1_text = "\n\n".join([d.page_content for d in docs[:2]])

            # Level 1: Create new vector store (handles cleanup of previous session)
            new_vectorstore = vector_manager.create_vector_store(docs)

            # Level 2: Reuse LLMService if it exists, just swap the vectorstore.
            # Create it fresh only on the very first upload.
            if llm_service is None:
                llm_service = LLMService(new_vectorstore, page1_text)
            else:
                llm_service.update_vectorstore(new_vectorstore, page1_text)
        
        # S3 background upload — runs after lock is released so it never
        # blocks the response. Fails silently if AWS is not configured.
        threading.Thread(
            target=_upload_to_s3_background,
            args=(temp_path, file.filename),
            daemon=True
        ).start()
        
        logger.info(f"Document '{file.filename}' uploaded and indexed successfully.")
        return jsonify({"message": "✅ Document Contextualized Successfully. You can now ask questions!"})

    except Exception as e:
        # FIX #14: Log the full error server-side; return a safe generic message to client.
        logger.error(f"Upload error for file '{file.filename}': {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while processing your document. Please try again."}), 500

    finally:
        # Always clean up the temp file regardless of success or failure
        if tmp_file and os.path.exists(tmp_file.name):
            try:
                os.remove(tmp_file.name)
            except Exception as e:
                logger.warning(f"Could not delete temp file '{tmp_file.name}': {e}")


@app.route('/chat', methods=['POST'])
def chat():
    if not llm_service:
        return jsonify({"answer": "Please upload a PDF document first to begin analysis."})

    # FIX #6: Safe JSON parsing — won't crash if body is missing or malformed
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"answer": "Invalid request format. Please send a JSON body."}), 400

    user_query = data.get("question", "").strip()
    if not user_query:
        return jsonify({"answer": "I didn't receive a question. Please type something and try again."})

    # Level 2: Delegate fully to the intelligence layer
    answer, context, page1 = llm_service.get_response(user_query)
    return jsonify({"answer": answer, "context": context, "page1": page1})


@app.route('/new_chat', methods=['POST'])
def new_chat():
    """
    FIX #13: This route is now wired to a button in the UI.
    Clears all session state and prepares the system for a new document.
    """
    global llm_service

    with _state_lock:
        vector_manager.delete_db()
        llm_service = None
        gc.collect()

    logger.info("Session cleared via /new_chat.")
    return jsonify({"message": "Session cleared. Please upload a new document to begin."})


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Validate environment variables before starting
    Config.validate()

    app.run(
        host='0.0.0.0',
        port=Config.PORT,
        debug=Config.DEBUG
    )