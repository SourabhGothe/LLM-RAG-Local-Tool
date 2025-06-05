# rag_utils.py
import os
import logging
from typing import List, Tuple, Optional, Dict

# Langchain components
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFium2Loader,  # Efficient PDF loader
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredFileLoader # Generic loader for other types if needed
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument # Alias to avoid confusion

# Import the Langchain-based embedding function from ollama_utils
from ollama_utils import get_ollama_embeddings, RAG_EMBEDDING_MODEL 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- RAG Configuration ---
CHROMA_PERSIST_DIR = os.path.join("data", "chroma_db_langchain") # New dir for Langchain version
COLLECTION_NAME = "ollama_lc_rag_collection"
NUM_RAG_RESULTS = 3  # Number of relevant chunks to retrieve for context
CHUNK_SIZE = 1000    # Target characters for RecursiveCharacterTextSplitter
CHUNK_OVERLAP = 150  # Overlap for RecursiveCharacterTextSplitter

# --- ChromaDB Client and Collection Initialization (Langchain way) ---
_chroma_vector_store: Optional[Chroma] = None

def get_vector_store() -> Optional[Chroma]:
    """Initializes and returns the Langchain Chroma vector store."""
    global _chroma_vector_store
    if _chroma_vector_store is None:
        if not os.path.exists(CHROMA_PERSIST_DIR):
            try:
                os.makedirs(CHROMA_PERSIST_DIR)
                logger.info(f"Created ChromaDB persistence directory: {CHROMA_PERSIST_DIR}")
            except OSError as e:
                logger.error(f"Failed to create ChromaDB directory {CHROMA_PERSIST_DIR}: {e}")
                raise RuntimeError(f"ChromaDB directory creation failed: {e}") from e
        
        embeddings = get_ollama_embeddings()
        if not embeddings:
            logger.error(f"Failed to get Ollama embeddings for RAG model '{RAG_EMBEDDING_MODEL}'. RAG will not function.")
            # Application should handle this, perhaps by disabling RAG features.
            raise RuntimeError(f"OllamaEmbeddings initialization failed for model '{RAG_EMBEDDING_MODEL}'.")

        try:
            _chroma_vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
            logger.info(f"Langchain Chroma vector store loaded/created. Collection: '{COLLECTION_NAME}', Path: '{CHROMA_PERSIST_DIR}'")
        except Exception as e:
            logger.error(f"Failed to initialize Langchain Chroma vector store: {e}")
            # This could be due to issues with the persist_directory or other Chroma/embedding errors.
            raise RuntimeError(f"Chroma vector store initialization failed: {e}") from e
            
    return _chroma_vector_store

# --- Document Loaders Map ---
# Using more specific loaders where possible.
LANGCHAIN_FILE_LOADERS = {
    '.pdf': PyPDFium2Loader,
    '.txt': TextLoader, # TextLoader expects encoding to be passed if not UTF-8
    '.docx': UnstructuredWordDocumentLoader,
    # Add more specific loaders if needed, e.g., for .csv, .html
    # '.html': UnstructuredHTMLLoader, # Example
    # '.csv': CSVLoader, # Example
}
# Fallback for other types if UnstructuredFileLoader is desired for broader support
USE_UNSTRUCTURED_FALLBACK = True


def load_documents_from_file(file_path: str, original_filename: str) -> Tuple[Optional[List[LangchainDocument]], Optional[str]]:
    """Loads documents from a file path using appropriate Langchain loader."""
    _, ext = os.path.splitext(original_filename)
    ext = ext.lower()

    loader_class = LANGCHAIN_FILE_LOADERS.get(ext)
    
    documents = []
    error_message = None

    try:
        if loader_class:
            if ext == '.txt': # TextLoader needs encoding specified or defaults to system default
                loader = loader_class(file_path, encoding='utf-8') # Assume UTF-8 for .txt
            else:
                loader = loader_class(file_path)
            documents = loader.load()
        elif USE_UNSTRUCTURED_FALLBACK:
            logger.info(f"Using UnstructuredFileLoader fallback for {original_filename} (extension: {ext})")
            # UnstructuredFileLoader can handle many types but might be slower or have more dependencies.
            # Ensure 'unstructured' and its sub-dependencies for specific file types are installed.
            loader = UnstructuredFileLoader(file_path, mode="single", strategy="fast") # or "hi_res" for PDFs
            documents = loader.load()
        else:
            error_message = f"Unsupported file type: {ext}. No specific Langchain loader configured and fallback is disabled."
            logger.warning(error_message)
            return None, error_message
        
        if not documents:
            error_message = f"No documents extracted from '{original_filename}' by the loader."
            logger.warning(error_message)
            return None, error_message

        logger.info(f"Successfully loaded {len(documents)} document(s) from '{original_filename}'.")
        return documents, None

    except Exception as e:
        error_message = f"Error loading file '{original_filename}' with Langchain loader: {e}"
        logger.error(error_message)
        return None, error_message


# --- Text Splitting ---
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Returns a configured Langchain RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False, # Use simple string separators
        separators=["\n\n", "\n", " ", ""] # Common separators
    )

# --- RAG Core Functions (Langchain based) ---
def add_content_to_rag(docs_to_add: List[LangchainDocument], source_name: str) -> str:
    """
    Splits Langchain documents and adds them to the Chroma vector store.
    Updates document metadata with the source.
    """
    if not docs_to_add:
        return f"No document content provided from '{source_name}' to add to RAG."

    try:
        vector_store = get_vector_store()
        if not vector_store:
            return "RAG vector store is not available. Cannot add content."
    except RuntimeError as e:
        return f"Failed to initialize RAG system: {e}"

    text_splitter = get_text_splitter()
    all_split_docs: List[LangchainDocument] = []
    for i, doc in enumerate(docs_to_add):
        # Update metadata before splitting to ensure it propagates
        doc.metadata["source"] = doc.metadata.get("source", source_name) # Preserve original source from loader if any, else use filename
        doc.metadata["original_doc_index"] = i 
        
        # Split the content of the document
        split_chunks_text = text_splitter.split_text(doc.page_content)
        
        for j, chunk_text in enumerate(split_chunks_text):
            # Create new LangchainDocument for each chunk, copying metadata
            # and adding chunk-specific info
            chunk_doc = LangchainDocument(
                page_content=chunk_text,
                metadata={**doc.metadata, "chunk_num_in_doc": j}
            )
            all_split_docs.append(chunk_doc)
            
    if not all_split_docs:
        logger.warning(f"Content from '{source_name}' resulted in no processable chunks after splitting.")
        return f"Content from '{source_name}' could not be chunked or was empty."

    try:
        # Generate IDs for Chroma, or let Chroma handle it (though explicit can be better for some cases)
        # For simplicity, let Chroma generate IDs if not specified.
        # If providing IDs, ensure they are unique. Example: [f"{source_name}_{i}" for i in range(len(all_split_docs))]
        vector_store.add_documents(all_split_docs)
        vector_store.persist() # Explicitly persist after adding
        logger.info(f"Added {len(all_split_docs)} chunks from '{source_name}' to Langchain Chroma RAG.")
        return f"Successfully added content from '{source_name}' ({len(all_split_docs)} chunks) to RAG."
    except Exception as e:
        logger.error(f"Error adding documents to Langchain Chroma for '{source_name}': {e}")
        # This could happen if embedding generation fails (e.g., Ollama server down, RAG_EMBEDDING_MODEL issue)
        return (f"Error adding content from '{source_name}' to RAG: {e}. "
                f"Ensure embedding model '{RAG_EMBEDDING_MODEL}' is functional in Ollama.")


def add_file_to_rag_langchain(file_path: str, original_filename: str) -> str:
    """Processes a file using Langchain loaders and adds its content to RAG."""
    
    # Load documents from file
    loaded_documents, error = load_documents_from_file(file_path, original_filename)
    if error or not loaded_documents:
        # Cleanup the uploaded file as it won't be processed
        if os.path.exists(file_path):
            try: os.remove(file_path); logger.info(f"Cleaned up unprocessed file: {file_path}")
            except OSError as e: logger.error(f"Error deleting unprocessed file {file_path}: {e}")
        return error or f"No content extracted from '{original_filename}'."

    # Add the loaded Langchain Documents to RAG
    # The source_name for metadata will be the original_filename
    result_message = add_content_to_rag(loaded_documents, source_name=original_filename)
    
    # Clean up the uploaded file after processing attempt
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up processed uploaded file: {file_path}")
        except OSError as e:
            logger.error(f"Error deleting processed uploaded file {file_path}: {e}")
            # Append to result message if cleanup fails, but don't override main message
            result_message += f" (Warning: failed to delete temp file {file_path})"
            
    return result_message


def add_text_to_rag_langchain(text_content: str, source_name: str = "direct_text_input") -> str:
    """Adds raw text to the RAG knowledge base using Langchain."""
    if not text_content.strip():
        return "No text content provided."

    # Create a Langchain Document object from the raw text
    # Metadata can include the source name
    doc = LangchainDocument(page_content=text_content, metadata={"source": source_name})
    
    return add_content_to_rag([doc], source_name=source_name)


def query_rag_langchain(prompt_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Queries the Langchain Chroma RAG for relevant context."""
    try:
        vector_store = get_vector_store()
        if not vector_store:
            return None, "RAG vector store is not available."
    except RuntimeError as e:
        return None, f"Failed to access RAG system: {e}"

    # Check if collection is empty (more involved with Chroma wrapper, direct client might be easier for count)
    # For now, let similarity_search handle it; it will return empty if no results.
    # A simple count:
    try:
        # The Langchain Chroma wrapper doesn't directly expose `count()`.
        # We might need to access the underlying client if a count is strictly needed here.
        # However, for query purposes, if `similarity_search` returns nothing, that's sufficient.
        # Example: `vector_store._collection.count()` if you need direct access (use with caution)
        pass 
    except Exception: # pylint: disable=bare-except
        pass


    logger.info(f"Querying RAG with prompt: '{prompt_text[:100]}...'")
    try:
        # Use similarity search to find relevant documents
        retrieved_docs: List[LangchainDocument] = vector_store.similarity_search(
            query=prompt_text,
            k=NUM_RAG_RESULTS
        )

        if retrieved_docs:
            context_parts = []
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get("source", "Unknown source")
                content_preview = doc.page_content.replace("\n", " ")[:150] # Short preview
                logger.info(f"Retrieved relevant chunk {i+1} from '{source}': '{content_preview}...'")
                context_parts.append(f"[Source: {source}]\n{doc.page_content}")
            
            context = "\n\n---\n\n".join(context_parts)
            logger.info(f"Retrieved {len(retrieved_docs)} context chunks for RAG.")
            return context, None
        else:
            logger.info("No relevant documents found in RAG for the prompt.")
            return None, None # No error, just no context
    except Exception as e:
        logger.error(f"Error querying Langchain Chroma RAG: {e}")
        return None, f"Error querying RAG: {e}. Ensure embedding model is functional."


def get_rag_status_langchain() -> Tuple[Dict, Optional[str]]:
    """Gets the status of the RAG collection (e.g., number of items)."""
    try:
        vector_store = get_vector_store()
        if not vector_store:
            return {"item_count": 0, "error": "RAG vector store not available."}, "RAG vector store not available."
        
        # Accessing the underlying Chroma client's collection to get count
        # This is a bit of an internal access, but often necessary for counts.
        count = vector_store._collection.count() # type: ignore
        return {"item_count": count}, None
    except RuntimeError as e:
         return {"item_count": 0, "error": f"RAG system (ChromaDB) init failed: {e}"}, str(e)
    except Exception as e:
        logger.error(f"Error getting RAG status from Langchain Chroma: {e}")
        return {"item_count": 0, "error": str(e)}, f"Error getting RAG status: {e}"


def clear_rag_collection_langchain() -> Tuple[bool, str]:
    """
    Clears all items from the RAG collection.
    For Chroma with Langchain, the most straightforward way to "clear" a persisted collection
    is often to delete its persistence directory and re-initialize, or delete/recreate the collection
    if the underlying client offers a clean method.
    The Langchain Chroma wrapper itself doesn't have a `clear_collection` method.
    We will delete and recreate the collection via the underlying client.
    """
    global _chroma_vector_store
    
    current_status, _ = get_rag_status_langchain()
    if current_status.get("item_count", 0) == 0 and "error" not in current_status:
        # If already empty and no errors, consider it cleared. Re-init to be safe.
        _chroma_vector_store = None # Force re-init on next get_vector_store()
        try:
            get_vector_store() # Re-initialize
            logger.info("RAG collection was already empty. Re-initialized.")
            return True, "RAG knowledge base was already empty. Re-initialized."
        except RuntimeError as e:
            return False, f"RAG was empty, but re-initialization failed: {e}"


    try:
        # Get the underlying client from the existing vector_store instance
        # This assumes get_vector_store() was successful at least once or can be called.
        temp_vs = get_vector_store() # Ensure it's initialized or error out here
        if not temp_vs:
             return False, "RAG vector store not available to clear."
        
        chroma_client = temp_vs._client # Access underlying client

        logger.info(f"Attempting to delete existing collection '{COLLECTION_NAME}' from path '{CHROMA_PERSIST_DIR}'.")
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            logger.info(f"Successfully deleted collection '{COLLECTION_NAME}'.")
        except ValueError as ve: # chromadb < 0.4.15 might raise ValueError for non-existent collection
            logger.warning(f"Collection '{COLLECTION_NAME}' might not have existed or another issue: {ve}")
            # Continue to re-creation
        except Exception as e_del: # Catch other potential errors during delete
             # chromadb >= 0.4.15 uses specific exceptions like CollectionNotFound, but generic for safety
            logger.warning(f"Could not definitively delete collection '{COLLECTION_NAME}' (it may not have existed): {e_del}")
            # Proceed to try and re-create it.

        # Force re-initialization of the vector store on next access
        _chroma_vector_store = None
        
        # Re-initialize to create a fresh collection
        get_vector_store() # This will call get_or_create_collection implicitly

        logger.info(f"RAG knowledge base (collection '{COLLECTION_NAME}') cleared and re-initialized successfully.")
        return True, "RAG knowledge base cleared and re-initialized successfully."

    except RuntimeError as e_rt: # Catch init errors from get_vector_store
        return False, f"Failed to access RAG system to clear: {e_rt}"
    except Exception as e:
        logger.error(f"An unexpected error occurred while clearing RAG collection: {e}")
        _chroma_vector_store = None # Attempt to reset on error
        return False, f"Error clearing RAG knowledge base: {e}. Attempting to reset."