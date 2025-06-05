# ollama_utils.py
import requests
import json
import logging
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OLLAMA_API_BASE_URL_FOR_ADMIN = "http://localhost:11434/api" # For admin tasks not covered by Langchain's Ollama class
OLLAMA_BASE_URL_FOR_LANGCHAIN = "http://localhost:11434" # Langchain's Ollama classes use the base URL

# This is the model that will be used by Langchain's OllamaEmbeddings
# Ensure this model is pulled in your Ollama instance (e.g., ollama pull nomic-embed-text)
RAG_EMBEDDING_MODEL = "nomic-embed-text" 

# --- Ollama Instance and Langchain Wrappers ---
_ollama_llm_instance_cache = {}
_ollama_embeddings_instance = None

def get_ollama_llm(model_name: str):
    """
    Returns a cached instance of Langchain's Ollama LLM for the given model.
    """
    global _ollama_llm_instance_cache
    if model_name not in _ollama_llm_instance_cache:
        try:
            # You can specify other parameters like temperature, top_k, etc.
            _ollama_llm_instance_cache[model_name] = Ollama(
                base_url=OLLAMA_BASE_URL_FOR_LANGCHAIN,
                model=model_name
            )
            logger.info(f"Initialized Langchain Ollama LLM for model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Langchain Ollama LLM for {model_name}: {e}")
            return None
    return _ollama_llm_instance_cache[model_name]

def get_ollama_embeddings():
    """
    Returns a cached instance of Langchain's OllamaEmbeddings.
    Uses the RAG_EMBEDDING_MODEL.
    """
    global _ollama_embeddings_instance
    if _ollama_embeddings_instance is None:
        if not RAG_EMBEDDING_MODEL:
            logger.error("RAG_EMBEDDING_MODEL is not set. Cannot initialize OllamaEmbeddings.")
            return None
        try:
            _ollama_embeddings_instance = OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL_FOR_LANGCHAIN,
                model=RAG_EMBEDDING_MODEL
            )
            logger.info(f"Initialized Langchain OllamaEmbeddings with model: {RAG_EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Langchain OllamaEmbeddings with model {RAG_EMBEDDING_MODEL}: {e}")
            # This could be due to Ollama server not running or model not available.
            return None
    return _ollama_embeddings_instance

# --- Ollama Server Admin Functions (Direct API calls) ---
def is_ollama_running():
    """Checks if the Ollama server is running and accessible."""
    try:
        # Langchain's Ollama constructor doesn't explicitly check server liveness beforehand in a simple way we can reuse here.
        # So, a direct check remains useful.
        response = requests.get(OLLAMA_BASE_URL_FOR_LANGCHAIN, timeout=3) # Check base Ollama URL
        if response.status_code == 200 and "Ollama is running" in response.text:
            return True
        # Try API endpoint as well
        requests.get(f"{OLLAMA_API_BASE_URL_FOR_ADMIN}/tags", timeout=3).raise_for_status()
        return True
    except requests.exceptions.RequestException:
        logger.warning("Ollama server is not responding (checked via direct requests).")
        return False

def get_available_models_admin():
    """Fetches the list of locally available Ollama models via direct API call."""
    if not is_ollama_running():
        return None, "Ollama server not responding. Please ensure Ollama is running."
    try:
        response = requests.get(f"{OLLAMA_API_BASE_URL_FOR_ADMIN}/tags")
        response.raise_for_status()
        models_data = response.json().get("models", [])
        return [model['name'] for model in models_data if model.get('name')], None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching models from Ollama (admin API): {e}")
        return None, f"Error fetching models (admin API): {e}"
    except json.JSONDecodeError:
        logger.error("Error decoding JSON response from Ollama when fetching models (admin API).")
        return None, "Error decoding response from Ollama (admin API)."

def pull_model_admin(model_name: str):
    """Pulls a model from Ollama Hub via direct API call."""
    if not is_ollama_running():
        return False, "Ollama server not responding."
    try:
        logger.info(f"Attempting to pull model via admin API: {model_name}")
        response = requests.post(
            f"{OLLAMA_API_BASE_URL_FOR_ADMIN}/pull",
            json={"name": model_name, "stream": False},
            timeout=600  # Extended timeout for model downloads
        )
        response.raise_for_status()
        # For stream=False, Ollama might return various statuses.
        # A 200 OK often means the request was accepted and is processing or completed.
        # Example success: {"status":"success"}
        # If model already exists, it might also return quickly.
        logger.info(f"Pull command for {model_name} sent. Status: {response.status_code}. Response: {response.text[:200]}") # Log snippet of response
        # A more reliable check would be to re-list models after a delay.
        # For now, we assume 200 means it's likely working or done.
        return True, f"Model '{model_name}' pull initiated/completed. Check Ollama logs or refresh model list."
    except requests.exceptions.Timeout:
        logger.error(f"Timeout when trying to pull model {model_name} (admin API).")
        return False, f"Timeout when trying to pull model '{model_name}'. It might still be downloading."
    except requests.exceptions.RequestException as e:
        error_detail = str(e)
        if e.response is not None:
            try:
                error_detail = e.response.json().get("error", e.response.text)
            except json.JSONDecodeError:
                error_detail = e.response.text
        logger.error(f"Error pulling model {model_name} (admin API): {error_detail}")
        return False, f"Error pulling model '{model_name}': {error_detail}"
    except Exception as e:
        logger.error(f"An unexpected error occurred while pulling model {model_name} (admin API): {e}")
        return False, f"An unexpected error occurred while pulling model '{model_name}'."

# --- Langchain-based Generation and Embedding Functions ---
def generate_response_langchain(model_name: str, prompt: str, context: str = None):
    """Generates a response using Langchain's Ollama LLM wrapper."""
    if not is_ollama_running(): # Good to have a quick check
        return None, "Ollama server not responding."

    llm = get_ollama_llm(model_name)
    if not llm:
        return None, f"Could not initialize LLM for model '{model_name}'. Ensure Ollama is running and the model is available."

    full_prompt = prompt
    if context:
        full_prompt = f"Using the following context retrieved from the knowledge base:\n---CONTEXT---\n{context}\n---END CONTEXT---\n\nBased on this context, please answer the following question or complete the task:\n{prompt}"
    
    logger.info(f"Generating response with Langchain Ollama model: {model_name}")
    try:
        response_text = llm.invoke(full_prompt)
        return response_text, None
    except Exception as e:
        logger.error(f"Error generating response from Langchain Ollama ({model_name}): {e}")
        # Langchain might wrap Ollama errors, try to give a meaningful message.
        return None, f"Error during generation with {model_name}: {e}. Ensure the model is pulled and Ollama server is stable."


def check_rag_embedding_model_availability_langchain():
    """
    Checks if the RAG_EMBEDDING_MODEL can be initialized via Langchain's OllamaEmbeddings.
    This implicitly checks if Ollama is running and the model is likely available.
    """
    if not is_ollama_running(): # Pre-check
        return False, "Ollama server not responding."
        
    logger.info(f"Checking availability of RAG embedding model '{RAG_EMBEDDING_MODEL}' via Langchain...")
    embeddings_instance = get_ollama_embeddings() # This attempts initialization
    
    if embeddings_instance:
        # Try a dummy embedding to be more certain
        try:
            embeddings_instance.embed_query("test")
            logger.info(f"RAG embedding model '{RAG_EMBEDDING_MODEL}' is available and functional via Langchain.")
            return True, f"RAG embedding model '{RAG_EMBEDDING_MODEL}' is available."
        except Exception as e:
            msg = (f"RAG embedding model '{RAG_EMBEDDING_MODEL}' found but failed a test embedding: {e}. "
                   "Ensure it's correctly pulled and functional in Ollama.")
            logger.warning(msg)
            return False, msg
    else:
        msg = (f"Failed to initialize RAG embedding model '{RAG_EMBEDDING_MODEL}' via Langchain. "
               f"Ensure Ollama is running and the model '{RAG_EMBEDDING_MODEL}' is pulled and valid for embeddings.")
        logger.warning(msg)
        return False, msg

