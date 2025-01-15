# Document Processing and Vector Search System

## Overview

This project is a Python wrapper around [Ollama](https://ollama.com/) and [ChromaDB](https://www.trychroma.com/), designed to leverage local, small-scale Language Models (LLMs) for **RAG (Retrieval-Augmented Generation)** processes. It offers a "democratic" approach to AI by utilizing multiple models for various tasks, enhancing the robustness and accuracy of information processing and retrieval.

**Note:** This is primarily a Proof of Concept (POC) with no frontend implementation. It's meant to be forked, customized, and extended to fit your specific needs.

### Key Features

- **RAG Implementation:** Enhances LLM outputs by retrieving relevant information from a vector database before generating responses.
- **Multiple Model Utilization:** Uses several smaller, specialized LLMs for processing and interpreting information.
- **Resource Efficiency:** Designed to run efficiently on systems without a GPU, though it can leverage one if available.
- **Modular Design:** Operates in three main modes:
  1. **PDF Processing:** Converts PDF documents to text using OCR and direct text extraction.
  2. **Vectorization:** Processes text into embeddings and stores them in ChromaDB for RAG.
  3. **Interactive Chat:** Allows for conversational interaction with the stored data using RAG.

### Customizability and Extensibility

- **Open for Contributions:** The codebase is structured to be easily customizable. You can add new models, modify processing steps, or integrate additional data sources.
- **Extendable Architecture:** The modular nature allows for the addition of new functionalities like different data persistence methods or additional interaction modes beyond chat and query.

### Directory Structure and Functionalities

#### `src/services/document/`:
- **pdf_processor.py:**
  - `VisionPDFProcessor`: Converts PDF documents into text using OCR for image-based content and direct text parsing for text-based PDFs. The initial step of OCR involves converting PDFs to images. After processing, these images are removed to save space. This is the initial step for RAG, preparing documents for embedding.

- **vector_processor.py:**
  - `VectorProcessor`: Converts processed text into vector embeddings, stored in ChromaDB, facilitating RAG by enabling efficient content retrieval based on similarity.

#### `src/optimization/`:
- **resources.py:**
  - `ResourceManager`: Manages system resources, monitoring CPU, memory, or disk usage, which is crucial for optimizing performance during large-scale data processing or model execution.

#### `src/helpers/`:
- **processing_helper.py:**
  - **ProcessingHelper:** Central utility for orchestrating document processing:
    - **Directory Management:** Ensures necessary directories exist for operation.
    - **Resource Management:** Initializes and cleans up resource monitoring for performance tracking.
    - **PDF Processing:** `process_pdfs_to_json` converts PDFs to JSON, using both OCR and text parsing.
    - **JSON to Vector Conversion:** `process_json_to_vector_db` processes JSON into embeddings for storage in ChromaDB, supporting RAG.
    - **Status Reporting:** Provides visual feedback on processing status.
    - **Metrics Export:** Exports system resource metrics for analysis or debugging.

  - **get_helper:** Factory function to instantiate `ProcessingHelper` with initialized resources.

### Document Input and Processing

- **Current Support:** The system currently only consumes PDF files.
- **PDF Input:** PDFs should be placed in the `files_to_process` directory for processing. After processing, the images created during the OCR step are deleted to clean up.

### How It Works

- **PDF Processing:** Combines OCR (first converting PDFs to images, then processing) with direct text extraction to get comprehensive document content for RAG.
- **Vector Embedding:** Text is encoded into vectors to be searchable and retrievable, key for RAG.
- **Querying and Chatting:** Uses RAG to provide context-aware responses based on document content.

### Resource Management

- **CPU/GPU Utilization:** Leverages multiprocessing for CPU and uses GPU through Ollama and PyTorch for model acceleration.
- **GPU Support:** Enhances RAG performance with GPU availability.

### Main Tools Used

- **Ollama:** Backend for running LLMs for RAG processes.
- **ChromaDB:** Vector storage for RAG.
- **Tesseract OCR:** For text extraction from images.
- **pdf2image:** PDF to image conversion for OCR.
- **PyMuPDF (fitz):** Direct PDF text extraction.
- **Rich:** Console output enhancement.
- **PyTorch:** Underlying for model execution in RAG.

### Getting Started

- **Python Virtual Environment:** Recommended for dependency management:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  pip install -r requirements.txt

### Processing a pdf
Note: Usually this takes some time, especially if you have large documents

![process-pdf](https://github.com/user-attachments/assets/b49ebf0c-36cc-4a70-8f37-8c0571984a37)


### Embedding the information in ChromaDB

![process-embedding](https://github.com/user-attachments/assets/b0459835-21fb-45cc-88c5-bf0ca425058e)

### Chat

![chat](https://github.com/user-attachments/assets/5387dd52-ec72-41ec-baea-272c91de1321)

You can do anything else as a normal LLM not only PDFs (RAG), although, keep in mind 1) the more documents in your database, the slower the chat, 2) it still can do mistakes, double check the answers.
