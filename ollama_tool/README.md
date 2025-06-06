# Ollama Multi-LLM Interaction Tool with RAG

This tool provides a web interface to interact with multiple Large Language Models (LLMs) hosted locally via Ollama. It includes features for model management (listing, pulling new models) and Retrieval Augmented Generation (RAG) by allowing you to upload documents or add text as a knowledge base.

## Features

* **Simple Web UI:** Easy-to-use interface for sending prompts and receiving responses.
* **Ollama Integration:** Leverages your local Ollama installation to run LLMs on your GPU.
* **Model Management:**
    * Lists currently available (pulled) Ollama models.
    * Allows you to pull new models from Ollama Hub.
* **Retrieval Augmented Generation (RAG):**
    * Upload PDF, TXT, or DOCX files to build a knowledge base.
    * Add raw text directly to the knowledge base.
    * Prompts are augmented with relevant information from this knowledge base before being sent to the LLM.
* **Persistent RAG Context:** The RAG knowledge base is stored locally using ChromaDB and persists across sessions.
* **Copy Response:** Easily copy the LLM's response.
* **Network Accessible:** Can be accessed from other devices on your local network.
* **Lightweight:** The web server itself is lightweight, with the heavy lifting (LLM inference, embeddings) done by Ollama.

## Prerequisites

1.  **Linux System with GPU:** As per your request for GPU usage.
2.  **Ollama Installed and Running:**
    * Download and install Ollama from [https://ollama.com/](https://ollama.com/).
    * Ensure the Ollama service is running. You can test this by opening a terminal and typing `ollama list`.
    * **Important for RAG:** You need an embedding model pulled in Ollama. This tool is configured to use `nomic-embed-text` by default for RAG. If you don't have it, pull it:
        ```bash
        ollama pull nomic-embed-text
        ```
    * Pull at least one generative LLM to start with, e.g.:
        ```bash
        ollama pull llama3:8b
        # or
        ollama pull mistral
        ```
3.  **Python 3.8+ and pip:**
    * Check your Python version: `python3 --version`
    * Ensure pip is installed.

4.  If you are getting chromadb hnswlib error install below
   sudo apt-get update && sudo apt-get install build-essential
   sudo apt install python3-dev
## Setup

1.  **Create Project Directory:**
    Create a directory for the tool, for example, `ollama_multi_llm_tool`. Place all the provided files (`app.py`, `ollama_utils.py`, `rag_utils.py`, `requirements.txt`, `static/`, `templates/`) into this directory.

2.  **Create Virtual Environment (Recommended):**
    ```bash
    cd ollama_multi_llm_tool
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create Data Directories:**
    Inside the `ollama_multi_llm_tool` directory, create a `data` directory, and within `data`, create `chroma_db` and `uploads` directories:
    ```bash
    mkdir -p data/chroma_db
    mkdir -p data/uploads
    ```
    These directories are used by ChromaDB for persistence and for storing uploaded files temporarily during processing.

## Running the Application

1.  **Start the Flask Server:**
    From the `ollama_multi_llm_tool` directory (with your virtual environment activated if you created one):
    ```bash
    python app.py
    ```

2.  **Access the UI:**
    Open your web browser and go to `http://127.0.0.1:5000`.
    If you want to access it from another device on your local network, Flask will typically show a message like `Running on http://0.0.0.0:5000/`. Use your machine's local IP address (e.g., `http://192.168.1.X:5000`).

## Using the Tool

### 1. Model Selection
* The "Select Model" dropdown will show LLMs currently available in your Ollama instance.
* If the list is empty or you want a new model, use the "Pull New Model" section.

### 2. Pulling New Models
* Enter the name of the model you want to download (e.g., `llama3:latest`, `mistral:7b`, `codellama:13b`). Refer to [Ollama Hub](https://ollama.com/library) for available models.
* Click "Download Model". The UI will show a "Downloading..." status. This can take time.
* Once downloaded, the model list will refresh automatically. If not, manually click "Refresh Model List".

### 3. Sending a Prompt
* Type your prompt into the "Enter your prompt" text area.
* Select the desired LLM from the dropdown.
* Click "Generate Response".
* The response will appear in the "Response" section. You can copy it using the "Copy Response" button.

### 4. Using RAG (Retrieval Augmented Generation)

The RAG system allows the LLM to use information from documents you provide.

* **Adding Files to RAG Context:**
    * Click "Choose File" under the "RAG - Add Knowledge" section.
    * Select a `.txt`, `.pdf`, or `.docx` file.
    * Click "Upload and Add to RAG". The file will be processed, and its content will be added to the vector knowledge base.
* **Adding Text to RAG Context:**
    * Type or paste text into the "Or add text directly" text area.
    * Click "Add Text to RAG".
* **RAG Status:**
    * The "RAG Context Status" will show the number of documents/text chunks currently in the knowledge base.
* **Clearing RAG Context:**
    * Click "Clear RAG Knowledge Base" to remove all items from the RAG vector store. This is useful if you want to start fresh with new documents.
* **How RAG Works with Prompts:**
    * When you click "Generate Response", if there's content in the RAG knowledge base, the system will:
        1.  Find the most relevant chunks of text from your uploaded documents/text based on your prompt.
        2.  Prepend this relevant text as context to your original prompt.
        3.  Send this augmented prompt to the selected LLM.
    * This helps the LLM answer questions based on the specific information you've provided.

## Stopping the Application

* Go to the terminal where `python app.py` is running and press `Ctrl+C`.

## Troubleshooting

* **"Ollama server not responding" or errors related to model fetching/generation:**
    * Ensure Ollama is installed correctly and the `ollama serve` command or service is running.
    * Verify network connectivity to Ollama (default: `http://localhost:11434`).
* **"Embedding model 'nomic-embed-text' not found":**
    * Pull the default embedding model: `ollama pull nomic-embed-text`. The RAG feature relies on this.
* **File upload issues:**
    * Ensure the `data/uploads` directory exists and the application has write permissions to it.
* **ChromaDB issues:**
    * Ensure the `data/chroma_db` directory exists and the application has write permissions to it.
* **Slow model downloads:** This is normal for large models. Check the terminal running `app.py` for Ollama's progress messages (if any are relayed). The UI will show a general "Downloading..." status.

## Customization

* **Ollama API URL:** If your Ollama runs on a different host or port, modify `OLLAMA_API_BASE_URL` in `app.py` and `ollama_utils.py`.
* **Default Embedding Model:** Change `RAG_EMBEDDING_MODEL` in `app.py` and `rag_utils.py` if you prefer a different Ollama embedding model (ensure it's pulled).
* **Number of RAG Chunks:** Modify `NUM_RAG_RESULTS` in `rag_utils.py` to control how many relevant chunks are retrieved.

## Disclaimer

This tool interacts with powerful LLMs. Always review generated content critically. Ensure you comply with the terms of use for any models you download and use via Ollama.
