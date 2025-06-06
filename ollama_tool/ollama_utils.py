# ollama_utils.py
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OLLAMA_API_BASE_URL = "http://localhost:11434/api" # Default Ollama API URL
RAG_EMBEDDING_MODEL = "nomic-embed-text" # Default model for generating embeddings for RAG

def is_ollama_running():
    """Checks if the Ollama server is running and accessible."""
    try:
        response = requests.get(OLLAMA_API_BASE_URL.replace("/api", ""), timeout=3) # Check base URL
        # Ollama's root path might return a simple "Ollama is running" or a 404
        # if /api/tags is a better check for actual API readiness.
        # For now, any response from the base URL is a good sign.
        # A more robust check would be to hit a specific benign endpoint like /api/tags (handled in get_available_models)
        if response.status_code == 200 or response.status_code == 404: # Ollama returns 404 for base if not a specific endpoint
             # Try to hit /api/tags to be more certain
            try:
                requests.get(f"{OLLAMA_API_BASE_URL}/tags", timeout=3)
                return True
            except requests.exceptions.RequestException:
                return False
        return False
    except requests.exceptions.RequestException:
        return False

def get_available_models():
    """Fetches the list of locally available Ollama models."""
    if not is_ollama_running():
        logger.error("Ollama server is not responding.")
        return None, "Ollama server not responding. Please ensure Ollama is running."

    try:
        response = requests.get(f"{OLLAMA_API_BASE_URL}/tags")
        response.raise_for_status()  # Raise an exception for HTTP errors
        models = response.json().get("models", [])
        # Filter out potential embedding models from the main generation list if desired,
        # but for now, show all. Users should know which models are for generation.
        return [model['name'] for model in models if model.get('name')], None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching models from Ollama: {e}")
        return None, f"Error fetching models: {e}"
    except json.JSONDecodeError:
        logger.error("Error decoding JSON response from Ollama when fetching models.")
        return None, "Error decoding response from Ollama."


def pull_model(model_name):
    """Pulls a model from Ollama Hub."""
    if not is_ollama_running():
        return False, "Ollama server not responding."

    try:
        logger.info(f"Attempting to pull model: {model_name}")
        # Ollama's pull API streams responses. We need to handle this.
        # For simplicity in this example, we'll make the request and assume it completes.
        # A more robust solution would handle the stream and provide progress.
        response = requests.post(
            f"{OLLAMA_API_BASE_URL}/pull",
            json={"name": model_name, "stream": False}, # Set stream to False for a single JSON response on completion
            timeout=600 # Extended timeout for model downloads
        )
        response.raise_for_status()

        # Check response status if stream=False
        # Typical success is status "success"
        # If the model was already pulled, it might also return quickly.
        # Ollama's /api/pull with stream=False returns a single JSON object per line for progress
        # and finally a status. If it's not streaming, it returns after completion.
        # The actual response structure can vary.
        # This check is simplified.
        # A common final status is {"status":"success"} if stream=False
        # If stream=True, you'd iterate response.iter_lines()

        if response.status_code == 200:
            # With stream=False, the final response might be a summary.
            # If the text is empty, it might indicate success, or that the model was already present.
            # Example success with stream=False might be: {"status":"success"} or just an empty 200 OK
            # if the model is already manifest.
            # We need to be careful here. For simplicity, we assume 200 OK is good.
            logger.info(f"Pull command for {model_name} sent. Check Ollama logs for progress.")
            # It might be better to check /api/tags again after a short delay to confirm.
            # For now, assume success on 200 OK.
            return True, f"Model '{model_name}' pull initiated. It should be available shortly if not already."
        else:
            logger.error(f"Error pulling model {model_name}: {response.text}")
            return False, f"Failed to pull model '{model_name}'. Status: {response.status_code}. Details: {response.text}"


    except requests.exceptions.Timeout:
        logger.error(f"Timeout when trying to pull model {model_name}.")
        return False, f"Timeout when trying to pull model '{model_name}'. It might still be downloading in the background."
    except requests.exceptions.RequestException as e:
        logger.error(f"Error pulling model {model_name}: {e}")
        return False, f"Error pulling model '{model_name}': {e}"
    except Exception as e: # Catch any other unexpected error
        logger.error(f"An unexpected error occurred while pulling model {model_name}: {e}")
        return False, f"An unexpected error occurred while pulling model '{model_name}'."


def generate_response(model_name, prompt, context=None):
    """Generates a response from the specified Ollama model."""
    if not is_ollama_running():
        return None, "Ollama server not responding."

    full_prompt = prompt
    if context:
        full_prompt = f"Using the following context:\n{context}\n\nNow answer the following question or complete the task:\n{prompt}"

    logger.info(f"Generating response with model: {model_name}")
    try:
        response = requests.post(
            f"{OLLAMA_API_BASE_URL}/generate",
            json={
                "model": model_name,
                "prompt": full_prompt,
                "stream": False  # Get the full response at once
            },
            timeout=300 # Timeout for generation
        )
        response.raise_for_status()
        # The response when stream=False is a JSON object with the full response
        response_data = response.json()
        return response_data.get("response", ""), None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error generating response from Ollama ({model_name}): {e}")
        # Try to parse error from Ollama if available
        error_detail = str(e)
        try:
            err_json = e.response.json()
            if "error" in err_json:
                error_detail = err_json["error"]
        except: # pylint: disable=bare-except
            pass # Keep original error if parsing fails
        return None, f"Error generating response: {error_detail}"
    except json.JSONDecodeError:
        logger.error("Error decoding JSON response from Ollama during generation.")
        return None, "Error decoding response from Ollama."

def get_embedding(text_chunk):
    """Generates an embedding for a text chunk using the specified RAG_EMBEDDING_MODEL."""
    if not is_ollama_running():
        logger.error("Ollama server is not responding for embedding generation.")
        return None, "Ollama server not responding."

    if not RAG_EMBEDDING_MODEL:
        logger.error("RAG_EMBEDDING_MODEL not configured.")
        return None, "Embedding model not configured."

    # First, check if the embedding model is available locally.
    # This check could be done once at startup or less frequently.
    # For now, it's implicit in the call to /api/embeddings.
    # If the model is not found, Ollama's /api/embeddings will return an error.

    try:
        response = requests.post(
            f"{OLLAMA_API_BASE_URL}/embeddings",
            json={
                "model": RAG_EMBEDDING_MODEL,
                "prompt": text_chunk
            },
            timeout=60 # Timeout for embedding
        )
        response.raise_for_status()
        embedding_data = response.json()
        if "embedding" in embedding_data:
            return embedding_data["embedding"], None
        else:
            logger.error(f"Ollama API did not return an embedding for model {RAG_EMBEDDING_MODEL}. Response: {embedding_data}")
            return None, f"Failed to get embedding. Response: {embedding_data.get('error', 'Unknown error from Ollama embeddings API')}"

    except requests.exceptions.RequestException as e:
        error_message = f"Error getting embedding from Ollama ({RAG_EMBEDDING_MODEL}): {e}"
        try: # Try to get more specific error from Ollama
            if e.response is not None:
                error_detail = e.response.json().get("error", str(e))
                if "model" in error_detail and "not found" in error_detail: # More specific error for missing model
                     error_message = (f"Embedding model '{RAG_EMBEDDING_MODEL}' not found in Ollama. "
                                     f"Please pull it first (e.g., `ollama pull {RAG_EMBEDDING_MODEL}`). Details: {error_detail}")
                else:
                    error_message = f"Ollama API error for embeddings ({RAG_EMBEDDING_MODEL}): {error_detail}"
        except: # pylint: disable=bare-except
            pass # Keep the original more generic error message
        logger.error(error_message)
        return None, error_message
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON response from Ollama embeddings API ({RAG_EMBEDDING_MODEL}).")
        return None, f"Error decoding response from Ollama embeddings API ({RAG_EMBEDDING_MODEL})."
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_embedding: {e}")
        return None, f"An unexpected error occurred while generating embedding: {e}"

def check_embedding_model_availability():
    """Checks if the configured RAG_EMBEDDING_MODEL is available in Ollama."""
    if not is_ollama_running():
        return False, "Ollama server not responding."
    
    models, err = get_available_models()
    if err:
        return False, err
    
    if RAG_EMBEDDING_MODEL in models:
        # Further check: try to get a dummy embedding to ensure it's operational
        # This is because just being in the tags list doesn't mean it's fully working for embeddings.
        # However, this adds an extra API call. For simplicity, we can trust the list for now.
        # Or, the get_embedding function will naturally fail if it's listed but not working.
        logger.info(f"Embedding model '{RAG_EMBEDDING_MODEL}' is available.")
        return True, f"Embedding model '{RAG_EMBEDDING_MODEL}' is available."
    else:
        msg = (f"RAG embedding model '{RAG_EMBEDDING_MODEL}' is not available in Ollama. "
               f"Please pull it first (e.g., `ollama pull {RAG_EMBEDDING_MODEL}`). "
               "RAG functionality will be limited until this model is available.")
        logger.warning(msg)
        return False, msg

