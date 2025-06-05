document.addEventListener('DOMContentLoaded', () => {
    const modelSelect = document.getElementById('model-select');
    const refreshModelsBtn = document.getElementById('refresh-models-btn');
    const newModelNameInput = document.getElementById('new-model-name');
    const pullModelBtn = document.getElementById('pull-model-btn');
    const pullModelStatus = document.getElementById('pull-model-status');

    const promptInput = document.getElementById('prompt-input');
    const useRagCheckbox = document.getElementById('use-rag-checkbox');
    const generateBtn = document.getElementById('generate-btn');
    const responseOutput = document.getElementById('response-output');
    const generateStatus = document.getElementById('generate-status');
    const copyResponseBtn = document.getElementById('copy-response-btn');

    const ragFileInput = document.getElementById('rag-file-input');
    const uploadRagFileBtn = document.getElementById('upload-rag-file-btn');
    const ragFileStatus = document.getElementById('rag-file-status');

    const ragTextInput = document.getElementById('rag-text-input');
    const addRagTextBtn = document.getElementById('add-rag-text-btn');
    const ragTextStatus = document.getElementById('rag-text-status');
    
    const ragContextStatus = document.getElementById('rag-context-status');
    const clearRagBtn = document.getElementById('clear-rag-btn');
    const ragClearStatus = document.getElementById('rag-clear-status');

    const ollamaStatusIndicator = document.getElementById('ollama-status-indicator');
    const ragModelNameIndicator = document.getElementById('rag-model-name-indicator');
    const ragModelStatusIndicator = document.getElementById('rag-model-status-indicator');
    const refreshStatusBtn = document.getElementById('refresh-status-btn');
    const ollamaApiUrlDisplay = document.getElementById('ollama-api-url-display');
    const chromaDbPathDisplay = document.getElementById('chroma-db-path-display');


    // --- Utility Functions ---
    function displayStatus(element, message, isError = false) {
        element.textContent = message;
        element.className = isError ? 'status-message error' : 'status-message success';
        if (message) {
            element.style.display = 'block';
        } else {
            element.style.display = 'none';
        }
    }

    // --- Initial Data Loading and Status Checks ---
    async function fetchModels() {
        displayStatus(pullModelStatus, 'Fetching models...', false);
        modelSelect.innerHTML = '<option value="">Loading models...</option>';
        try {
            const response = await fetch('/api/models');
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: "Failed to fetch models. Server returned an error." }));
                throw new Error(errorData.error || `HTTP error ${response.status}`);
            }
            const data = await response.json();
            modelSelect.innerHTML = ''; // Clear loading/error message
            if (data.models && data.models.length > 0) {
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                });
            } else {
                modelSelect.innerHTML = '<option value="">No models found. Pull a model.</option>';
            }
            displayStatus(pullModelStatus, '', false); // Clear status
        } catch (error) {
            console.error('Error fetching models:', error);
            modelSelect.innerHTML = '<option value="">Error loading models</option>';
            displayStatus(pullModelStatus, `Error: ${error.message}`, true);
        }
    }

    async function fetchRagStatus() {
        try {
            const response = await fetch('/api/rag/status');
             if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: "Failed to fetch RAG status." }));
                throw new Error(errorData.error || `HTTP error ${response.status}`);
            }
            const data = await response.json();
            if (data.rag_status) {
                 ragContextStatus.textContent = `Items: ${data.rag_status.item_count || 0}`;
                 if(data.rag_status.error){
                    ragContextStatus.textContent += ` (Warning: ${data.rag_status.error})`;
                 }
            } else if (data.error) {
                 ragContextStatus.textContent = `Error: ${data.error}`;
            }
        } catch (error) {
            console.error('Error fetching RAG status:', error);
            ragContextStatus.textContent = `Error fetching status: ${error.message}`;
        }
    }
    
    async function checkSystemStatus() {
        ollamaStatusIndicator.textContent = "Checking...";
        ragModelStatusIndicator.textContent = "Checking...";
        ragModelNameIndicator.textContent = "N/A";
        ollamaApiUrlDisplay.textContent = "N/A";
        chromaDbPathDisplay.textContent = "N/A";

        try {
            const response = await fetch('/api/status');
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            const data = await response.json();

            ollamaStatusIndicator.textContent = data.ollama_running ? 'Running' : 'Not Running/Reachable';
            ollamaStatusIndicator.className = data.ollama_running ? 'status-ok' : 'status-error';
            
            ragModelNameIndicator.textContent = data.rag_embedding_model_configured || "N/A";
            ragModelStatusIndicator.textContent = data.rag_embedding_model_available ? 'Available' : 'Not Available';
            ragModelStatusIndicator.className = data.rag_embedding_model_available ? 'status-ok' : 'status-error';
            if (!data.rag_embedding_model_available && data.rag_embedding_model_status_message) {
                 ragModelStatusIndicator.textContent += ` (${data.rag_embedding_model_status_message})`;
            }


            ollamaApiUrlDisplay.textContent = data.ollama_api_url || "N/A";
            chromaDbPathDisplay.textContent = data.chroma_db_path || "N/A";

        } catch (error) {
            console.error('Error checking system status:', error);
            ollamaStatusIndicator.textContent = 'Error';
            ollamaStatusIndicator.className = 'status-error';
            ragModelStatusIndicator.textContent = 'Error';
            ragModelStatusIndicator.className = 'status-error';
        }
    }


    // --- Event Listeners ---
    refreshModelsBtn.addEventListener('click', fetchModels);

    pullModelBtn.addEventListener('click', async () => {
        const modelName = newModelNameInput.value.trim();
        if (!modelName) {
            displayStatus(pullModelStatus, 'Please enter a model name.', true);
            return;
        }
        displayStatus(pullModelStatus, `Downloading ${modelName}... This may take a while.`, false);
        pullModelBtn.disabled = true;
        try {
            const response = await fetch('/api/pull_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: modelName })
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || `HTTP error ${response.status}`);
            }
            displayStatus(pullModelStatus, data.message || `Model ${modelName} pull initiated.`, false);
            newModelNameInput.value = '';
            if (data.models) { // If server returns updated model list
                modelSelect.innerHTML = '';
                 if (data.models && data.models.length > 0) {
                    data.models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = model;
                        modelSelect.appendChild(option);
                    });
                } else {
                    modelSelect.innerHTML = '<option value="">No models found. Pull a model.</option>';
                }
            } else { // Fallback to manual refresh if not included
                fetchModels(); // Refresh model list
            }
             if(data.warning){
                displayStatus(pullModelStatus, `${pullModelStatus.textContent} Warning: ${data.warning}`, false); // Append warning
            }
        } catch (error) {
            console.error('Error pulling model:', error);
            displayStatus(pullModelStatus, `Error: ${error.message}`, true);
        } finally {
            pullModelBtn.disabled = false;
        }
    });

    generateBtn.addEventListener('click', async () => {
        const selectedModel = modelSelect.value;
        const currentPrompt = promptInput.value.trim();
        const shouldUseRag = useRagCheckbox.checked;

        if (!selectedModel) {
            displayStatus(generateStatus, 'Please select a model.', true);
            return;
        }
        if (!currentPrompt) {
            displayStatus(generateStatus, 'Please enter a prompt.', true);
            return;
        }

        displayStatus(generateStatus, 'Generating response...', false);
        responseOutput.innerHTML = '<p>Generating...</p>';
        copyResponseBtn.style.display = 'none';
        generateBtn.disabled = true;

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    model_name: selectedModel, 
                    prompt: currentPrompt,
                    use_rag: shouldUseRag 
                })
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || `HTTP error ${response.status}`);
            }
            responseOutput.textContent = data.response || "No response content.";
            displayStatus(generateStatus, data.rag_error_if_any ? `Response generated. RAG warning: ${data.rag_error_if_any}` : 'Response generated successfully.', false);
            copyResponseBtn.style.display = 'inline-block';
        } catch (error) {
            console.error('Error generating response:', error);
            responseOutput.textContent = `Error: ${error.message}`;
            displayStatus(generateStatus, `Error: ${error.message}`, true);
        } finally {
            generateBtn.disabled = false;
        }
    });
    
    copyResponseBtn.addEventListener('click', () => {
        const textToCopy = responseOutput.textContent;
        // navigator.clipboard.writeText is preferred but might not work in all iframe contexts
        // Using document.execCommand as a fallback
        try {
            const textArea = document.createElement('textarea');
            textArea.value = textToCopy;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            displayStatus(generateStatus, 'Response copied to clipboard!', false);
            setTimeout(() => displayStatus(generateStatus, '', false), 2000);
        } catch (err) {
            console.error('Failed to copy response:', err);
            displayStatus(generateStatus, 'Failed to copy response.', true);
        }
    });


    uploadRagFileBtn.addEventListener('click', async () => {
        const file = ragFileInput.files[0];
        if (!file) {
            displayStatus(ragFileStatus, 'Please select a file.', true);
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        displayStatus(ragFileStatus, `Uploading ${file.name}...`, false);
        uploadRagFileBtn.disabled = true;

        try {
            const response = await fetch('/api/rag/add_file', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
             if (!response.ok && !data.message) { // If not OK and no message, assume server error
                throw new Error(data.error || `HTTP error ${response.status}`);
            }
            // Even if response is not 200, data.message might contain a user-friendly error from backend
            const isError = response.status >= 400 || (data.message && data.message.toLowerCase().includes("error"));
            displayStatus(ragFileStatus, data.message || "File processed.", isError);
            
            if (data.rag_status) {
                ragContextStatus.textContent = `Items: ${data.rag_status.item_count || 0}`;
                if(data.rag_status.error){
                    ragContextStatus.textContent += ` (Warning: ${data.rag_status.error})`;
                 }
            } else if (data.rag_status_error) {
                 ragContextStatus.textContent += ` (RAG Status Update Error: ${data.rag_status_error})`;
            } else {
                fetchRagStatus(); // Fallback to refresh RAG status
            }
            ragFileInput.value = ''; // Clear file input
        } catch (error) {
            console.error('Error uploading RAG file:', error);
            displayStatus(ragFileStatus, `Error: ${error.message}`, true);
        } finally {
            uploadRagFileBtn.disabled = false;
        }
    });

    addRagTextBtn.addEventListener('click', async () => {
        const textContent = ragTextInput.value.trim();
        if (!textContent) {
            displayStatus(ragTextStatus, 'Please enter some text.', true);
            return;
        }
        displayStatus(ragTextStatus, 'Adding text to RAG...', false);
        addRagTextBtn.disabled = true;

        try {
            const response = await fetch('/api/rag/add_text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text_content: textContent })
            });
            const data = await response.json();
            const isError = response.status >= 400 || (data.message && data.message.toLowerCase().includes("error"));
            displayStatus(ragTextStatus, data.message || "Text added.", isError);
            
            if (data.rag_status) {
                ragContextStatus.textContent = `Items: ${data.rag_status.item_count || 0}`;
                 if(data.rag_status.error){
                    ragContextStatus.textContent += ` (Warning: ${data.rag_status.error})`;
                 }
            } else if (data.rag_status_error) {
                 ragContextStatus.textContent += ` (RAG Status Update Error: ${data.rag_status_error})`;
            } else {
                fetchRagStatus(); // Fallback refresh
            }
            ragTextInput.value = ''; // Clear text area
        } catch (error) {
            console.error('Error adding RAG text:', error);
            displayStatus(ragTextStatus, `Error: ${error.message}`, true);
        } finally {
            addRagTextBtn.disabled = false;
        }
    });
    
    clearRagBtn.addEventListener('click', async () => {
        if (!confirm("Are you sure you want to clear the entire RAG knowledge base? This cannot be undone.")) {
            return;
        }
        displayStatus(ragClearStatus, 'Clearing RAG knowledge base...', false);
        clearRagBtn.disabled = true;
        try {
            const response = await fetch('/api/rag/clear', { method: 'POST' });
            const data = await response.json();
            const isError = !data.success || (data.message && data.message.toLowerCase().includes("error"));
            displayStatus(ragClearStatus, data.message || "RAG cleared.", isError);

            if (data.rag_status) {
                ragContextStatus.textContent = `Items: ${data.rag_status.item_count || 0}`;
                 if(data.rag_status.error){
                    ragContextStatus.textContent += ` (Warning: ${data.rag_status.error})`;
                 }
            } else if (data.rag_status_error) {
                 ragContextStatus.textContent += ` (RAG Status Update Error: ${data.rag_status_error})`;
            } else {
                fetchRagStatus(); // Fallback refresh
            }
        } catch (error) {
            console.error('Error clearing RAG:', error);
            displayStatus(ragClearStatus, `Error: ${error.message}`, true);
        } finally {
            clearRagBtn.disabled = false;
        }
    });

    refreshStatusBtn.addEventListener('click', checkSystemStatus);


    // --- Initial Load ---
    checkSystemStatus();
    fetchModels();
    fetchRagStatus();
});