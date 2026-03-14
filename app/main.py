import os, gc
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

    # 1. Clear references to release memory locks
    vectorstore = None
    llm_service = None
    gc.collect() 

    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    
    try:
        file.save(temp_path)
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        # Create new vector store in a brand new unique folder
        vectorstore = vector_manager.create_vector_store(docs)
        llm_service = LLMService(vectorstore)
        
        return jsonify({"message": "✅ Session Initialized: Ready for Analysis"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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