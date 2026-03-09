import os
import shutil
import tempfile
from flask import Flask, render_template, request, jsonify
from app.config import Config
from app.service.storage_service import StorageService
from app.service.llm_service import LLMService
from app.models.vector_store import VectorStoreManager
from langchain_community.document_loaders import PyPDFLoader

app = Flask(__name__)

# Initialize components
storage_service = StorageService()
vector_manager = VectorStoreManager()

# Global variables for state management
vectorstore = None
llm_service = None

# Optional: Load existing vector store on startup
try:
    vectorstore = vector_manager.load_vector_store()
    if vectorstore:
        llm_service = LLMService(vectorstore)
except Exception as e:
    print(f"No existing vector store found: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global vectorstore, llm_service
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 1. Clear existing objects to release file locks on 'chroma_db'
    vectorstore = None
    llm_service = None
    
    # 2. Delete the old DB folder safely
    if os.path.exists("chroma_db"):
        try:
            shutil.rmtree("chroma_db")
            print("🧹 Old Knowledge Base cleared.")
        except Exception as e:
            print(f"Warning: Could not clear directory: {e}")

    # 3. Create temp path manually (Windows-safe)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save and upload
        file.save(temp_path)
        storage_service.upload_file(temp_path, file.filename)
        
        # Load and process PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        if len(docs) == 0:
            return jsonify({"error": "No text found in PDF."}), 400

        # 4. Re-initialize the Vector Store and LLM Service
        vectorstore = vector_manager.create_vector_store(docs)
        llm_service = LLMService(vectorstore)
        
        return jsonify({"message": "✅ Knowledge Base Updated!"})
    
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
        
    finally:
        # 5. Final cleanup of the temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except PermissionError:
                print(f"Wait: Could not delete {temp_path} - file still in use.")

@app.route('/chat', methods=['POST'])
def chat():
    if not llm_service:
        return jsonify({"answer": "Please upload a document first."})
    
    data = request.json
    user_query = data.get("question")
    answer = llm_service.get_response(user_query)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.PORT, debug=True)