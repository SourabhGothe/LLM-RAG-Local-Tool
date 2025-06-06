# app.py
import os
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Import utility functions
from ollama_utils import (
    get_available_models,
    pull_model,
    generate_response,
    is_ollama_running,
    check_embedding_model_availability,
    OLLAMA_API_BASE_URL, # For display/debug if needed
    RAG_EMBEDDING_MODEL
)
from rag_utils import (
    add_file_to_rag,
    add_text_to_rag,
    query_rag,
    get_rag_status,
    clear_rag_collection,
    FILE_READERS, # For allowed extensions
    CHROMA_PERSIST_DIR # For info
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('data', 'uploads')
ALLOWED_EXTENSIONS = set(FILE_READERS.keys())
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created upload folder: {UPLOAD_FOLDER}")

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \           os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def api_status():
    """Checks the status of Ollama and the RAG embedding model."""
    ollama_ok = is_ollama_running()
    rag_model_available, rag_model_message = False, "Ollama not running or RAG model check failed."

    if ollama_ok:
        rag_model_available, rag_model_message = check_embedding_model_availability()

    return jsonify({
        "ollama_running": ollama_ok,
        "ollama_api_url": OLLAMA_API_BASE_URL,
        "rag_embedding_model_configured": RAG_EMBEDDING_MODEL,
        "rag_embedding_model_available": rag_model_available,
        "rag_embedding_model_status_message": rag_model_message,
        "chroma_db_path": CHROMA_PERSIST_DIR
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """API endpoint to get the list of available Ollama models."""
    models, error = get_available_models()
    if error:
        return jsonify({"error": error}), 500
    return jsonify({"models": models})

@app.route('/api/pull_model', methods=['POST'])
def pull_new_model():
    """API endpoint to pull a new model via Ollama."""
    data = request.get_json()
    model_name = data.get('model_name')

    if not model_name:
        return jsonify({"error": "Model name is required."}), 400

    success, message = pull_model(model_name)
    if not success:
        return jsonify({"error": message}), 500
    
    # Fetch updated model list to send back
    models, err = get_available_models()
    if err: # If fetching models fails after pull, still return success for pull itself
        return jsonify({"message": message, "models": [], "warning": f"Pull initiated but failed to refresh model list: {err}"})
        
    return jsonify({"message": message, "models": models})


@app.route('/api/generate', methods=['POST'])
def generate():
    """API endpoint to generate a response from an Ollama model, with RAG."""
    data = request.get_json()
    model_name = data.get('model_name')
    prompt = data.get('prompt')
    use_rag = data.get('use_rag', True) # Default to using RAG if available

    if not model_name or not prompt:
        return jsonify({"error": "Model name and prompt are required."}), 400

    context = None
    rag_error = None
    if use_rag:
        # Check RAG status first (is it initialized and has items?)
        rag_status, rag_err_status = get_rag_status()
        if rag_err_status:
            logger.warning(f"RAG status check failed: {rag_err_status}")
            # Don't block generation, but log it.
        
        if rag_status and rag_status.get("item_count", 0) > 0 :
            context, rag_error = query_rag(prompt)
            if rag_error:
                logger.error(f"Error querying RAG: {rag_error}")
                # Proceed without RAG context if querying fails
                context = None 
        else:
            logger.info("RAG is enabled but knowledge base is empty or not initialized properly.")
            
    response_text, error = generate_response(model_name, prompt, context=context)

    if error:
        # Include RAG error information if it occurred
        full_error_message = error
        if rag_error:
            full_error_message = f"{error} (RAG query also failed: {rag_error})"
        return jsonify({"error": full_error_message}), 500
    
    return jsonify({"response": response_text, "rag_error_if_any": rag_error})


@app.route('/api/rag/add_file', methods=['POST'])
def upload_rag_file():
    """API endpoint to upload a file and add its content to RAG."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            logger.info(f"File '{filename}' saved to '{filepath}' for RAG processing.")
        except Exception as e:
            logger.error(f"Error saving uploaded file {filename}: {e}")
            return jsonify({"error": f"Could not save file: {e}"}), 500

        # The add_file_to_rag function will handle reading, chunking, embedding, and cleanup.
        message = add_file_to_rag(filepath, filename) # filepath is used for reading, filename for source_name

        # Update RAG status after adding
        status, err = get_rag_status()
        if err:
            return jsonify({"message": message, "rag_status_error": err}), 500 if "Error" in message else 200
            
        return jsonify({"message": message, "rag_status": status})
    else:
        return jsonify({"error": f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400


@app.route('/api/rag/add_text', methods=['POST'])
def add_rag_text():
    """API endpoint to add raw text to the RAG knowledge base."""
    data = request.get_json()
    text_content = data.get('text_content')

    if not text_content or not text_content.strip():
        return jsonify({"error": "Text content is required and cannot be empty."}), 400

    source_name = data.get('source_name', 'direct_text_input')
    message = add_text_to_rag(text_content, source_name)

    # Update RAG status after adding
    status, err = get_rag_status()
    if err:
        return jsonify({"message": message, "rag_status_error": err}), 500 if "Error" in message else 200
        
    return jsonify({"message": message, "rag_status": status})


@app.route('/api/rag/status', methods=['GET'])
def rag_collection_status():
    """API endpoint to get the current status of the RAG collection."""
    status, error = get_rag_status()
    if error: # This error is if the status function itself failed critically
        return jsonify({"error": error, "rag_status": {"item_count": 0, "error": error}}), 500
    # status might contain an "error" field if ChromaDB isn't initialized, but item_count will be 0.
    # This is handled by the client.
    return jsonify({"rag_status": status})

@app.route('/api/rag/clear', methods=['POST'])
def clear_rag():
    """API endpoint to clear the RAG knowledge base."""
    success, message = clear_rag_collection()
    
    # Get updated status
    status, err = get_rag_status()
    if err:
         return jsonify({"message": message, "rag_status_error": err, "success": success}), 500 if not success else 200

    return jsonify({"message": message, "rag_status": status, "success": success})


if __name__ == '__main__':
    # Ensure necessary directories exist (though rag_utils also tries for chroma)
    if not os.path.exists(CHROMA_PERSIST_DIR):
        os.makedirs(CHROMA_PERSIST_DIR)
        logger.info(f"Created ChromaDB persistence directory: {CHROMA_PERSIST_DIR}")

    logger.info("Starting Ollama Multi-LLM Tool Flask server.")
    logger.info(f"Ollama API configured at: {OLLAMA_API_BASE_URL}")
    logger.info(f"RAG Embedding Model configured as: {RAG_EMBEDDING_MODEL}")
    logger.info(f"ChromaDB will persist to: {CHROMA_PERSIST_DIR}")
    logger.info(f"File uploads will be temporarily stored in: {app.config['UPLOAD_FOLDER']}")
    
    # Initial check for Ollama and embedding model
    if not is_ollama_running():
        logger.warning("CRITICAL: Ollama server does not seem to be running or accessible. The application may not function correctly.")
    else:
        logger.info("Ollama server detected.")
        _, rag_model_msg = check_embedding_model_availability()
        logger.info(f"RAG Embedding Model Status: {rag_model_msg}")


    app.run(host='0.0.0.0', port=5000, debug=True) # debug=False for production