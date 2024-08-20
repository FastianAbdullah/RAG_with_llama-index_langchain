# Synthesia: Chat with Your Data

Synthesia is an efficient agentic RAG (Retrieval-Augmented Generation) system that combines the power of LlamaIndex and LangChain to provide an interactive chat experience with your documents.

## Features

-  Chat with AI without document context
-  Upload and process various document types (PDF, TXT, DOCX, XLSX)
-  Chat with AI using the context from uploaded documents
-  Efficient document processing and chunking
-  Utilizes Groq's LLM for fast responses
-  Interactive Streamlit interface

## How It Works

1. **Document Processing**: 
   - Upload documents (PDF, TXT, DOCX, XLSX)
   - Documents are chunked and stored in a Chroma vector database

2. **Chat Functionality**:
   - Chat with AI without document context
   - Chat with AI using the context from uploaded documents

3. **Query Routing**:
   - Uses LlamaIndex's RouterQueryEngine to efficiently route queries
   - Combines summary and vector indexing for comprehensive responses

## Setup
1. Clone the repository:
   ```
   https://github.com/FastianAbdullah/RAG_with_llama-index_langchain.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
## Usage:
   To start the application, run:
   ```
   streamlit run app.py
   ```

Then, navigate to the provided local URL in your web browser.

## Mermaid Flow Diagram

```mermaid
graph TD
    A[Start] --> B{Choose Functionality}
    B -->|Chat| C[Handle Chat]
    B -->|Chat with Documents| D[Handle Chat with Documents]
    B -->|Add Document| E[Handle Add Document]
    
    C --> F[Display Chat Messages]
    C --> G[Handle User Input]
    
    D --> H{Document Added?}
    H -->|Yes| I[Load Documents]
    H -->|No| J[Display Warning]
    I --> F
    I --> G
    
    E --> K[Upload File]
    K --> L[Process Document]
    L --> M[Add to Chroma DB]
    
    G --> N{Use Documents?}
    N -->|Yes| O[Query Router]
    N -->|No| P[LLM Response]
    
    O --> Q[Summary Index]
    O --> R[Vector Index]
    Q --> S[Generate Response]
    R --> S
    
    P --> S
    S --> T[Display Response]
