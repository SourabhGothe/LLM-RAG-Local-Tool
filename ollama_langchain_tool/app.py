# app.py
import os
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Import LANGCHAIN-BASED utility functions
from ollama_utils import (
    # LLM and Embeddings - Langchain based
    generate_response_langchain,
    check_rag_embedding_model_availability_langchain,
    # Admin functions - Direct Ollama API
    get_available_models_admin,
    pull_model_admin,
    is_ollama_running,
    # Constants
    OLLAMA_BASE_URL_FOR_LANGCHAIN, # For display/debug
    RAG_EMBEDDING_MODEL
)
from rag_utils import (
    # RAG functions - Langchain based
    add_file_to_rag_langchain,
    add_text_to_rag_langchain,
    query_rag_langchain,
    get_rag_status_langchain,
    clear_rag_collection_langchain,
    # Constants
    LANGCHAIN_FILE_LOADERS, # For allowed extensions info
    CHROMA_PERSIST_DIR, # For info
    USE_UNSTRUCTURED_FALLBACK
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('data', 'uploads_langchain') # Separate upload folder
# Determine allowed extensions from Langchain loaders configuration
ALLOWED_EXTENSIONS = set(LANGCHAIN_FILE_LOADERS.keys())
if USE_UNSTRUCTURED_FALLBACK:
    # If using UnstructuredFileLoader as a fallback, it supports many types.
    # For simplicity, we might not list them all or just rely on backend validation.
    # For now, stick to explicitly defined ones for the primary check.
    pass 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created upload folder: {UPLOAD_FOLDER}")

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def api_status():
    """Checks the status of Ollama and the RAG embedding model using Langchain."""
    ollama_ok = is_ollama_running() # Direct check for Ollama server
    rag_model_available, rag_model_message = False, "Ollama not running or RAG model check failed."

    if ollama_ok:
        # Use the Langchain-based check for the embedding model
        rag_model_available, rag_model_message = check_rag_embedding_model_availability_langchain()

    return jsonify({
        "ollama_running": ollama_ok,
        "ollama_api_url": OLLAMA_BASE_URL_FOR_LANGCHAIN, # Display Langchain-compatible base URL
        "rag_embedding_model_configured": RAG_EMBEDDING_MODEL,
        "rag_embedding_model_available": rag_model_available,
        "rag_embedding_model_status_message": rag_model_message,
        "chroma_db_path": CHROMA_PERSIST_DIR # Display Langchain Chroma path
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """API endpoint to get the list of available Ollama models (uses admin direct call)."""
    models, error = get_available_models_admin() # Using the direct admin call
    if error:
        return jsonify({"error": error}), 500
    return jsonify({"models": models or []}) # Ensure models is a list

@app.route('/api/pull_model', methods=['POST'])
def pull_new_model():
    """API endpoint to pull a new model via Ollama (uses admin direct call)."""
    data = request.get_json()
    model_name = data.get('model_name')

    if not model_name:
        return jsonify({"error": "Model name is required."}), 400

    success, message = pull_model_admin(model_name) # Using the direct admin call
    if not success:
        return jsonify({"error": message}), 500
    
    models, err = get_available_models_admin()
    if err:
        return jsonify({"message": message, "models": [], "warning": f"Pull task reported: '{message}'. Failed to refresh model list: {err}"})
        
    return jsonify({"message": message, "models": models or []})


@app.route('/api/generate', methods=['POST'])
def generate():
    """API endpoint to generate a response from an Ollama model, with Langchain RAG."""
    data = request.get_json()
    model_name = data.get('model_name')
    prompt = data.get('prompt')
    use_rag = data.get('use_rag', True) 

    if not model_name or not prompt:
        return jsonify({"error": "Model name and prompt are required."}), 400

    context_str: Optional[str] = None
    rag_query_error: Optional[str] = None

    if use_rag:
        rag_status_data, rag_status_err = get_rag_status_langchain()
        if rag_status_err:
            logger.warning(f"RAG status check failed: {rag_status_err}")
            # Proceed, but this might indicate RAG system issues.
        
        if rag_status_data and rag_status_data.get("item_count", 0) > 0 and not rag_status_data.get("error"):
            context_str, rag_query_error = query_rag_langchain(prompt)
            if rag_query_error:
                logger.error(f"Error querying RAG with Langchain: {rag_query_error}")
                # Proceed without RAG context if querying fails
        elif rag_status_data.get("error"):
             logger.warning(f"RAG system has an error state: {rag_status_data.get('error')}. Skipping RAG for this query.")
        else:
            logger.info("RAG is enabled but knowledge base is empty or reported 0 items.")
            
    # Use the Langchain-based generation function
    response_text, error = generate_response_langchain(model_name, prompt, context=context_str)

    if error:
        full_error_message = error
        if rag_query_error: # If RAG also had an issue, append it
            full_error_message = f"LLM Error: {error} (RAG query also failed: {rag_query_error})"
        return jsonify({"error": full_error_message}), 500
    
    return jsonify({"response": response_text, "rag_error_if_any": rag_query_error})


@app.route('/api/rag/add_file', methods=['POST'])
def upload_rag_file():
    """API endpoint to upload a file and add its content to RAG using Langchain."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    original_filename = secure_filename(file.filename)
    if file and allowed_file(original_filename): # Use original_filename for extension check
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename) # Save with original (secured) name
        
        try:
            file.save(filepath)
            logger.info(f"File '{original_filename}' saved to '{filepath}' for Langchain RAG processing.")
        except Exception as e:
            logger.error(f"Error saving uploaded file {original_filename}: {e}")
            return jsonify({"error": f"Could not save file: {e}"}), 500

        # Use Langchain-based function (it handles reading, chunking, embedding, cleanup)
        message = add_file_to_rag_langchain(filepath, original_filename) 

        status_data, status_err = get_rag_status_langchain()
        response_payload = {"message": message, "rag_status": status_data}
        if status_err:
            response_payload["rag_status_error"] = status_err
        
        # Determine status code based on whether the core operation (message) indicates an error
        http_status_code = 500 if "error" in message.lower() or "failed" in message.lower() else 200
        return jsonify(response_payload), http_status_code
    else:
        allowed_types_str = ', '.join(ALLOWED_EXTENSIONS) if ALLOWED_EXTENSIONS else "None configured"
        return jsonify({"error": f"File type not allowed for '{original_filename}'. Allowed: {allowed_types_str}"}), 400


@app.route('/api/rag/add_text', methods=['POST'])
def add_rag_text():
    """API endpoint to add raw text to the RAG knowledge base using Langchain."""
    data = request.get_json()
    text_content = data.get('text_content')

    if not text_content or not text_content.strip():
        return jsonify({"error": "Text content is required and cannot be empty."}), 400

    source_name = data.get('source_name', 'direct_text_input')
    # Use Langchain-based function
    message = add_text_to_rag_langchain(text_content, source_name)

    status_data, status_err = get_rag_status_langchain()
    response_payload = {"message": message, "rag_status": status_data}
    if status_err:
            response_payload["rag_status_error"] = status_err

    http_status_code = 500 if "error" in message.lower() or "failed" in message.lower() else 200
    return jsonify(response_payload), http_status_code


@app.route('/api/rag/status', methods=['GET'])
def rag_collection_status():
    """API endpoint to get the current status of the RAG collection (Langchain based)."""
    status_data, error_msg = get_rag_status_langchain()
    # error_msg here is if the status function itself failed critically.
    # status_data might contain an "error" field from deeper RAG issues (e.g., DB not init).
    if error_msg and "error" not in status_data: # If top-level error_msg but no error in status_data itself
        status_data["error"] = error_msg # Ensure an error is reported

    # The client-side JS already handles cases where rag_status.error exists.
    return jsonify({"rag_status": status_data}), 200 if not status_data.get("error") else 500


@app.route('/api/rag/clear', methods=['POST'])
def clear_rag():
    """API endpoint to clear the RAG knowledge base (Langchain based)."""
    success, message = clear_rag_collection_langchain()
    
    status_data, status_err = get_rag_status_langchain() # Get updated status
    response_payload = {"message": message, "rag_status": status_data, "success": success}
    if status_err :
         response_payload["rag_status_error"] = status_err
    
    http_status_code = 200 if success else 500
    return jsonify(response_payload), http_status_code


if __name__ == '__main__':
    # Ensure necessary directories exist
    if not os.path.exists(CHROMA_PERSIST_DIR): # For Langchain Chroma
        try:
            os.makedirs(CHROMA_PERSIST_DIR)
            logger.info(f"Created ChromaDB (Langchain) persistence directory: {CHROMA_PERSIST_DIR}")
        except OSError as e:
            logger.error(f"FATAL: Could not create ChromaDB directory {CHROMA_PERSIST_DIR}: {e}")
            # Consider exiting if this essential directory can't be made.
            exit(1)

    logger.info("Starting Ollama Multi-LLM Tool (Langchain Enhanced) Flask server.")
    logger.info(f"Ollama API (for Langchain) configured at: {OLLAMA_BASE_URL_FOR_LANGCHAIN}")
    logger.info(f"RAG Embedding Model (for Langchain) configured as: {RAG_EMBEDDING_MODEL}")
    logger.info(f"Langchain ChromaDB will persist to: {CHROMA_PERSIST_DIR}")
    logger.info(f"File uploads will be temporarily stored in: {app.config['UPLOAD_FOLDER']}")
    
    if not is_ollama_running():
        logger.warning("CRITICAL: Ollama server does not seem to be running or accessible via direct check. Langchain components might fail.")
    else:
        logger.info("Ollama server detected by direct check.")
        # Perform Langchain-based check for embedding model readiness
        is_ready, rag_model_msg = check_rag_embedding_model_availability_langchain()
        if is_ready:
            logger.info(f"Langchain RAG Embedding Model Status: {rag_model_msg}")
        else:
            logger.warning(f"Langchain RAG Embedding Model Status: {rag_model_msg} - RAG functionality might be impaired.")
    
    # Pre-warm RAG system by trying to get vector store instance once at startup
    try:
        from rag_utils import get_vector_store
        get_vector_store()
        logger.info("Successfully pre-initialized Langchain RAG vector store.")
    except RuntimeError as e:
        logger.error(f"Failed to pre-initialize Langchain RAG vector store at startup: {e}. RAG features may fail.")
    except Exception as e_gen:
        logger.error(f"Unexpected error during RAG vector store pre-initialization: {e_gen}")


    app.run(host='0.0.0.0', port=5000, debug=True) # debug=False for production
