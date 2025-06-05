# Ollama Multi-LLM Interaction Tool with Langchain RAG

This tool provides a web interface to interact with multiple Large Language Models (LLMs) hosted locally via Ollama. It has been **enhanced with Langchain** for more robust and flexible Retrieval Augmented Generation (RAG) capabilities and LLM interactions.

## Features

* **Simple Web UI:** Easy-to-use interface for sending prompts and receiving responses.
* **Ollama Integration via Langchain:**
    * Leverages `langchain-community` for interactions with your local Ollama LLMs and Embedding models.
    * Direct Ollama API calls are still used for administrative tasks like listing and pulling models.
* **Model Management:**
    * Lists currently available (pulled) Ollama models.
    * Allows you to pull new models from Ollama Hub.
* **Langchain-Powered Retrieval Augmented Generation (RAG):**
    * Upload PDF, TXT, or DOCX files to build a knowledge base.
    * Uses Langchain's `DocumentLoaders` for file parsing.
    * Employs Langchain's `RecursiveCharacterTextSplitter` for text chunking.
    * Uses `OllamaEmbeddings` (via Langchain) for creating text embeddings with a local Ollama model (default: `nomic-embed-text`).
    * Stores embeddings and text chunks in a `Chroma` vector database (via Langchain).
    * Add raw text directly to the knowledge base.
    * Prompts are augmented with relevant information retrieved by Langchain from this knowledge base.
* **Persistent RAG Context:** The RAG knowledge base (ChromaDB) persists across sessions in `data/chroma_db_langchain/`.
* **Copy Response:** Easily copy the LLM's response.
* **Network Accessible:** Can be accessed from other devices on your local network.
* **Lightweight Server:** The Flask web server itself remains lightweight.

## Prerequisites

1.  **Linux System with GPU:** (Recommended for Ollama GPU acceleration).
2.  **Ollama Installed and Running:**
    * Download and install Ollama from [https://ollama.com/](https://ollama.com/).
    * Ensure the Ollama service is running (e.g., `ollama serve` or system service). Test with `ollama list`.
    * **Crucial for RAG:** You **must** have an embedding model pulled in Ollama. This tool is configured to use `nomic-embed-text` by default for RAG via Langchain. Pull it if you haven't:
        ```bash
        ollama pull nomic-embed-text
        ```
    * Pull at least one generative LLM, e.g.:
        ```bash
        ollama pull llama3:8b
        # or
        ollama pull mistral
        ```
3.  **Python 3.8+ and pip:**
    * Check version: `python3 --version`.
    * Ensure pip is installed.

## Setup

1.  **Create Project Directory:**
    Create a directory (e.g., `ollama_langchain_tool`). Place all provided files (`app.py`, `ollama_utils.py`, `rag_utils.py`, `requirements.txt`, `static/`, `templates/`) into this directory.

2.  **Create Virtual Environment (Highly Recommended):**
    ```bash
    cd ollama_langchain_tool
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install Flask, Requests, Langchain packages, ChromaDB, document loaders, etc.

4.  **Create Data Directories:**
    Inside your project directory (`ollama_langchain_tool`), create the necessary `data` subdirectories:
    ```bash
    mkdir -p data/chroma_db_langchain 
    mkdir -p data/uploads_langchain
    ```
    * `data/chroma_db_langchain/`: For ChromaDB persistence (Langchain version).
    * `data/uploads_langchain/`: For temporary storage of uploaded files.

## Running the Application

1.  **Start the Flask Server:**
    From your project directory (with virtual environment activated):
    ```bash
    python app.py
    ```
    The server will typically run on `http://0.0.0.0:5000/`.

2.  **Access the UI:**
    Open your web browser and go to `http://127.0.0.1:5000`.
    To access from other devices on your local network, use your machine's local IP address (e.g., `http://192.168.X.Y:5000`).

## Using the Tool

The UI and basic operations (model selection, pulling models, sending prompts) remain largely the same as the previous version.

### Key Langchain Enhancements (Mostly Backend)

* **Document Handling:** When you upload files or add text for RAG, Langchain's document loaders and text splitters process the content more robustly.
* **Embeddings:** `OllamaEmbeddings` from Langchain generates embeddings using your local Ollama `nomic-embed-text` (or configured) model.
* **Vector Storage:** `Chroma` (via Langchain) manages the vector store.
* **Retrieval:** When "Use RAG" is checked, Langchain's Chroma integration is used to find relevant documents to augment the prompt.

### RAG Operations
* **Adding Knowledge:** Upload files (PDF, TXT, DOCX) or paste text. Langchain handles the processing and storage into the Chroma vector DB.
* **RAG Status:** Shows the number of items (chunks) in the Langchain ChromaDB.
* **Clearing RAG:** Removes the current RAG collection and re-initializes it.

## Stopping the Application

* In the terminal running `python app.py`, press `Ctrl+C`.

## Troubleshooting

* **"Ollama server not responding" / "Failed to initialize Langchain Ollama...":**
    * Ensure Ollama is installed and the service is running (`ollama serve`).
    * Verify the `OLLAMA_BASE_URL_FOR_LANGCHAIN` in `ollama_utils.py` (default `http://localhost:11434`) is correct.
* **"RAG embedding model 'nomic-embed-text' not available/functional" or similar errors:**
    * Make **sure** you have pulled the model: `ollama pull nomic-embed-text`.
    * Check Ollama server logs for issues with this model.
    * The application status banner provides feedback on embedding model availability.
* **File upload/processing errors:**
    * Ensure the `data/uploads_langchain` directory exists and has write permissions.
    * Some file types (especially complex PDFs or DOCX) might require additional system libraries for `unstructured` or `pypdfium2` to work correctly. If you encounter issues with specific files, check the console logs from `app.py` for detailed error messages from Langchain loaders.
    * The `requirements.txt` includes common loaders. For very exotic formats, `unstructured[all-docs]` might be needed, but it's a larger install.
* **ChromaDB issues:**
    * Ensure `data/chroma_db_langchain` directory exists and has write permissions.
    * If you see errors like "SQLite connectable is already closed", it might indicate an issue with ChromaDB's persistence or concurrent access (though this app is single-process). Restarting the app might help. If persistent, deleting the `chroma_db_langchain` directory (and losing stored RAG data) and restarting the app will create a fresh database.

## Customization

* **Ollama URLs:**
    * `OLLAMA_BASE_URL_FOR_LANGCHAIN` in `ollama_utils.py` for Langchain interactions.
    * `OLLAMA_API_BASE_URL_FOR_ADMIN` in `ollama_utils.py` for direct admin API calls.
* **RAG Embedding Model:** Change `RAG_EMBEDDING_MODEL` in `ollama_utils.py` (and ensure the new model is pulled into Ollama and suitable for embeddings).
* **RAG Parameters:** `CHUNK_SIZE`, `CHUNK_OVERLAP`, `NUM_RAG_RESULTS` in `rag_utils.py` can be tuned.
* **Document Loaders:** To support more file types for RAG, you can add more Langchain document loaders in `rag_utils.py` (in `LANGCHAIN_FILE_LOADERS`) and ensure necessary supporting libraries are in `requirements.txt`.

This Langchain-enhanced version provides a more robust and extensible foundation for your local LLM interaction tool.
