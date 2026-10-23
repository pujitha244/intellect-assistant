 Intellect Assistant – Advanced Local RAG System

**A privacy-focused AI assistant for document analysis and question answering using Retrieval-Augmented Generation (RAG).**

🔗 **GitHub Repository:** https://github.com/pujitha244/intellect-assistant

## Overview

Intellect Assistant is a document-based question-answering system that allows users to upload personal documents and ask questions about their contents. It retrieves relevant information from uploaded documents and uses a locally hosted Large Language Model (LLM) to generate contextual answers with source citations.

The project combines semantic search, keyword-based retrieval, document processing, vector databases, and conversational memory to provide an interactive knowledge assistant.

## Features

- **Hybrid Search:** Combines semantic similarity search with keyword-based retrieval to improve relevant document retrieval.
- **Conversational Memory:** Maintains context across questions and supports follow-up queries.
- **Adaptive Chunking:** Uses document-aware chunking strategies to preserve useful context.
- **Multi-format Document Support:** Supports PDF, DOCX, TXT, and Markdown files, subject to the implemented file handlers.
- **Local LLM Integration:** Uses Ollama with Llama 3.2 for locally hosted language-model inference.
- **Source Attribution:** Provides document references to help users verify generated answers.
- **Interactive Web Interface:** Provides a Streamlit interface for uploading documents and asking questions.
- **Privacy-focused Processing:** Designed to process documents and run inference locally, subject to configured components and any external integrations.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core application logic |
| Streamlit | Interactive web interface |
| LangChain | RAG pipeline and language-model integration |
| ChromaDB | Vector storage and similarity search |
| Sentence Transformers | Text embeddings |
| Scikit-learn | TF-IDF and keyword-based retrieval utilities |
| Ollama | Local language-model runtime |
| Llama 3.2 | Answer generation |

## System Architecture

The application follows a Retrieval-Augmented Generation workflow.

1. **Document ingestion:** Users upload supported documents through the Streamlit interface.
2. **Text extraction:** The document processor extracts text from uploaded files.
3. **Chunking:** Extracted text is divided into smaller, overlapping chunks.
4. **Embedding and indexing:** Text chunks are converted into embeddings and stored in ChromaDB.
5. **Hybrid retrieval:** Semantic similarity and keyword-based search identify relevant chunks.
6. **Context construction:** Retrieved passages are assembled into context for the language model.
7. **Answer generation:** Ollama runs the configured local LLM to generate a contextual response.
8. **Source attribution and memory:** The application presents supporting document references and manages conversation context.

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- Ollama installed on your system
- Sufficient RAM and disk space for the selected embedding and language models

### 1. Clone the repository

```bash
git clone https://github.com/pujitha244/intellect-assistant.git
cd intellect-assistant
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation reports a missing dependency, verify that it is listed in `requirements.txt` and compatible with your Python version.

### 4. Install and configure Ollama

Download Ollama from:

https://ollama.com/download

Pull the configured model:

```bash
ollama pull llama3.2:1b
```

Ensure Ollama is running and that the model name matches the application's configuration.

### 5. Run the application

From the project root, execute:

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal, typically:

```text
http://localhost:8501
```

Open the URL in your browser to access the application.

## How to Use

1. Launch the application.
2. Upload a supported document through the sidebar.
3. Click **Process Documents** to extract, chunk, embed, and index the content.
4. Enter a question about the uploaded documents.
5. Review the generated answer and its supporting source references.
6. Ask follow-up questions to explore the documents using conversation context.
7. Compare information across multiple documents where supported by the implementation.

## Project Structure

```text
intellect-assistant/
├── app.py
├── config.py
├── requirements.txt
├── .gitignore
├── src/
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── qa_chain.py
│   └── conversation_manager.py
└── data/
    ├── uploads/
    └── chroma_db/
```

*Note: This is the intended logical structure. Actual filenames and directories should be checked against the current repository.*

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface and application workflow |
| `config.py` | Model and retrieval configuration |
| `document_processor.py` | Document extraction and chunking |
| `vector_store.py` | Vector storage and retrieval logic |
| `qa_chain.py` | Context construction and question answering |
| `conversation_manager.py` | Conversation history and contextual follow-ups |
| `requirements.txt` | Python dependencies |

## Key Technical Concepts

### Retrieval-Augmented Generation (RAG)

RAG grounds an LLM's response in information retrieved from a knowledge source. Rather than relying only on the model's pretrained knowledge, the application retrieves relevant document passages and supplies them as context for answer generation.

### Hybrid Search

Hybrid retrieval combines two complementary approaches:

- **Semantic search:** Finds passages with similar meanings using vector embeddings.
- **Keyword search:** Finds passages containing relevant terms using techniques such as TF-IDF.

Combining their results can improve retrieval when a question contains either conceptual language or specific keywords.

### Conversational Memory

Conversation management retains relevant earlier exchanges so follow-up questions can be interpreted in context. For example, a user can ask a second question referring to a document discussed in the previous turn.

### Local Language Models

Ollama enables the application to run a supported language model on the user's machine. This reduces reliance on external LLM inference APIs, although privacy also depends on the application's other integrations and configuration.

## Configuration

The following settings may be configured in `config.py`, depending on the current implementation:

| Setting | Description |
|---|---|
| `CHUNK_SIZE` | Maximum chunk size |
| `CHUNK_OVERLAP` | Overlap between adjacent chunks |
| `LLM_MODEL` | Ollama model identifier |
| `MAX_RETRIEVED_CHUNKS` | Maximum number of chunks retrieved for a query |

Check the actual configuration file for the supported names and default values.

## Limitations and Considerations

- Answer quality depends on document extraction, chunking, retrieval relevance, and model capability.
- Local inference performance depends on available hardware and model size.
- Source references should be checked against the original documents.
- Large documents may require additional memory, storage, and processing time.
- The application should be tested with a clean environment and sample documents before deployment.

## Future Enhancements

Potential improvements include:

- Automated evaluation of retrieval precision and answer relevance.
- Improved ranking and reranking of retrieved passages.
- Expanded document-format support.
- Document-level access controls and deletion workflows.
- More comprehensive automated tests and error handling.
- A containerized deployment option.

## Author and Repository

**Project:** Intellect Assistant – Advanced Local RAG System

**GitHub:** https://github.com/pujitha244/intellect-assistant

Developed using Python, Streamlit, LangChain, ChromaDB, Sentence Transformers, and Ollama.

---
