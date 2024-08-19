import streamlit as st
from dotenv import load_dotenv
import os

# Langchain Imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document as LangChainDocument

# Other imports
import pandas as pd
import docx2txt
import chromadb
from chromadb.config import Settings

# LlamaIndex imports
from llama_index.core import Settings as LlamaSettings, SummaryIndex, VectorStoreIndex, Document as LlamaDocument
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector


# Load environment variables
load_dotenv()

groq_api_key = st.secrets["GROQ_API_KEY"]
user_agent = st.secrets["USER_AGENT"]

if groq_api_key is None:
    raise ValueError("Groq API key not found. Please set the GROQ_API_KEY environment variable in your .env file.")

# Initialize components
folder_path = "db"
chat_history = []
cached_llm = ChatGroq(model_name="llama3-8b-8192")
embedding = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# LlamaIndex settings
LlamaSettings.llm = Groq(api_key=groq_api_key, model="llama3-8b-8192")
LlamaSettings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

raw_prompt = PromptTemplate(
    input_variables=["input", "context"],
    template="""
    You are a helpful assistant that provides information based on the given context. 
    If the information is not in the context, politely say that you don't have that information.
    
    Context: {context}
    
    Human: {input}
    Assistant: """
)

class DataLoader:
    def __init__(self, filepath_or_url, user_agent=None):
        self.filepath_or_url = filepath_or_url
        self.user_agent = user_agent
    
    def load_document(self):
        file_extension = os.path.splitext(self.filepath_or_url)[1].lower()
        
        if file_extension == '.pdf':
            loader = PDFPlumberLoader(self.filepath_or_url)
            documents = loader.load()
        elif file_extension == '.txt':
            with open(self.filepath_or_url, 'r', encoding='utf-8') as file:
                text = file.read()
            documents = [LangChainDocument(page_content=text, metadata={"source": self.filepath_or_url})]
        elif file_extension in ['.docx', '.doc']:
            text = docx2txt.process(self.filepath_or_url)
            documents = [LangChainDocument(page_content=text, metadata={"source": self.filepath_or_url})]
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(self.filepath_or_url)
            documents = [LangChainDocument(page_content=df.to_string(), metadata={"source": self.filepath_or_url})]
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return documents

    def chunk_document(self, documents, chunk_size=1024, chunk_overlap=80):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

    def process_document(self, chunk_size=1024, chunk_overlap=80):
        documents = self.load_document()
        if len(documents) == 1 and self.filepath_or_url.endswith(('.xlsx', '.xls')):
            chunks = documents
        else:
            chunks = self.chunk_document(documents, chunk_size, chunk_overlap)
        print(f"Number of chunks: {len(chunks)}")
       
        return chunks

def convert_to_llama_documents(data):
    llama_documents = []
    if isinstance(data, dict) and 'documents' in data:  # Chroma data
        for i, doc_text in enumerate(data['documents']):
            metadata = data['metadatas'][i] if i < len(data['metadatas']) else {}
            llama_doc = LlamaDocument(
                text=doc_text,
                metadata=metadata
            )
            llama_documents.append(llama_doc)
    elif isinstance(data, list):
        for doc in data:
            if isinstance(doc, LangChainDocument):
                llama_doc = LlamaDocument(
                    text=doc.page_content,
                    metadata=doc.metadata
                )
            elif isinstance(doc, LlamaDocument):
                llama_doc = doc
            else:
                raise ValueError(f"Unsupported document type: {type(doc)}")
            llama_documents.append(llama_doc)
    else:
        raise ValueError("Unsupported data format")
    return llama_documents

def create_query_router(documents):
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    vector_query_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description="Useful for summarization questions"
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description="Useful for retrieving specific context"
    )

    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )

    return query_engine

def get_chatbot_response(query, chat_history, documents=None):
    if documents:
        llama_documents = convert_to_llama_documents(documents)
        query_engine = create_query_router(llama_documents)
        response = query_engine.query(query)
        return str(response)
    else:
        chat_history_str = ''.join([f"{m['role'].capitalize()}: {m['content']}\n" for m in chat_history])
        full_prompt = f"Chat History:\n{chat_history_str}Human: {query}\nAI:"
        response = cached_llm.invoke([HumanMessage(content=full_prompt)])
        return response.content
    
def initialize_chroma_client():
    return chromadb.PersistentClient(path=folder_path)
    chromadb.Per

def main():
    # Create necessary folders
    os.makedirs("document_storage", exist_ok=True)
    os.makedirs(folder_path, exist_ok=True)

    if 'chroma_client' not in st.session_state:
        st.session_state.chroma_client = initialize_chroma_client()

    st.title("Synthesia, Chat with your Data. Efficient Agentic RAG with Mix Blend of Llama-index and Langchain")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_added" not in st.session_state:
        st.session_state.document_added = False

    option = st.sidebar.selectbox(
        "Choose a functionality",
        ("Chat", "Chat with Documents", "Add Document")
    )

    if option == "Chat":
        handle_chat()
    elif option == "Chat with Documents":
        handle_chat_with_documents()
    elif option == "Add Document":
        handle_add_document()

def handle_chat():
    display_chat_messages()
    handle_user_input(use_documents=False)

def handle_chat_with_documents():
    if not st.session_state.document_added:
        st.warning("Please add a document before chatting with documents.")
        return

    collection = st.session_state.chroma_client.get_or_create_collection(name="my_collection")
    documents = collection.get()
    if not documents['documents']:
        st.warning("No document content available. Please add a non-empty document.")
        return
    display_chat_messages()
    handle_user_input(use_documents=True, documents=documents)

def handle_add_document():
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt', 'docx', 'xlsx'])
    if uploaded_file is not None:
        with st.spinner('Processing...'):
            try:
                # Ensure the document_storage directory exists
                os.makedirs("document_storage", exist_ok=True)
                
                save_path = os.path.join("document_storage", uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                data_loader = DataLoader(save_path, user_agent)
                chunks = data_loader.process_document(chunk_size=1024, chunk_overlap=80)

                if not chunks:
                    st.error(f"No content could be extracted from {uploaded_file.name}. Please ensure the file is not empty and try again.")
                    st.session_state.document_added = False
                    return

                # Initialize Chroma
                # chroma_client = chromadb.Client(Settings(persist_directory=folder_path, anonymized_telemetry=False))
                collection = st.session_state.chroma_client.get_or_create_collection(name="my_collection")

                # Add documents to the collection
                collection.add(
                    documents=[chunk.page_content for chunk in chunks],
                    metadatas=[chunk.metadata for chunk in chunks],
                    ids=[f"id_{i}" for i in range(len(chunks))]
                )

                st.session_state.document_added = True
                st.success(f"Successfully uploaded {uploaded_file.name} and processed {len(chunks)} chunk(s).")
            except Exception as e:
                st.error(f"An error occurred while processing the document: {str(e)}")
                st.session_state.document_added = False

def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(use_documents=False, documents=None):
    prompt = st.chat_input("What is your question?")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = get_chatbot_response(prompt, st.session_state.messages, documents if use_documents else None)

        if not response or response.strip() == "Empty Response":
            response = "I apologize, but I couldn't find any relevant information to answer your question. Could you please rephrase your question or ask about something else related to the uploaded documents?"

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()