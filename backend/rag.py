import os
import shutil
import json
import time
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

WORKSPACES_ROOT = "./data/workspaces"

# Ensure root workspace dir exists
if not os.path.exists(WORKSPACES_ROOT):
    os.makedirs(WORKSPACES_ROOT, exist_ok=True)


class RAGService:
    def __init__(self):
        self.current_embedding_model = "nomic-embed-text"  # Default fallback
        self.vector_store_cache = {}

    def _get_user_dir(self, user_id: int) -> str:
        return os.path.join(WORKSPACES_ROOT, str(user_id))

    def get_workspace_path(self, user_id: int, workspace_name: str) -> str:
        return os.path.join(self._get_user_dir(user_id), workspace_name)

    def get_chroma_path(self, user_id: int, workspace_name: str) -> str:
        return os.path.join(self.get_workspace_path(user_id, workspace_name), "chroma_db")

    def get_file_metadata_path(self, user_id: int, workspace_name: str) -> str:
        return os.path.join(self.get_workspace_path(user_id, workspace_name), "file_metadata.json")

    def save_file_metadata(self, user_id: int, workspace_name: str, filename: str, chunk_count: int):
        path = self.get_file_metadata_path(user_id, workspace_name)
        data = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except:
                pass
        
        data[filename] = {
            "chunk_count": chunk_count,
            "upload_timestamp": time.time()
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def get_workspace_files(self, user_id: int, workspace_name: str) -> Dict[str, Any]:
        """Get list of files and metadata for workspace."""
        path = self.get_file_metadata_path(user_id, workspace_name)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def create_workspace(self, user_id: int, workspace_name: str) -> bool:
        """Create a new workspace directory structure for a specific user."""
        path = self.get_workspace_path(user_id, workspace_name)
        if os.path.exists(path):
            raise ValueError(f"Workspace '{workspace_name}' already exists.")
        
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, "uploads"), exist_ok=True)
        return True

    def list_workspaces(self, user_id: int) -> List[str]:
        """List all available workspaces for a user."""
        user_dir = self._get_user_dir(user_id)
        if not os.path.exists(user_dir):
            return []
        
        workspaces = [
            d for d in os.listdir(user_dir) 
            if os.path.isdir(os.path.join(user_dir, d)) and not d.startswith(".")
        ]
        return sorted(workspaces)

    def delete_workspace(self, user_id: int, workspace_name: str) -> bool:
        """Delete a workspace and its data."""
        path = self.get_workspace_path(user_id, workspace_name)
        if os.path.exists(path):
            shutil.rmtree(path)
            # Remove from cache if present
            cache_key = f"{user_id}_{workspace_name}"
            if cache_key in self.vector_store_cache:
                del self.vector_store_cache[cache_key]
            return True
        return False

    def get_vector_store(self, user_id: int, workspace_name: str, embedding_model: str):
        """Get or create a Chroma vector store instance for a workspace."""
        chroma_path = self.get_chroma_path(user_id, workspace_name)
        
        embeddings = OllamaEmbeddings(model=embedding_model)
        
        vector_store = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )
        return vector_store

    def process_file(self, user_id: int, workspace_name: str, file_path: str, embedding_model: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        """Process a file (PDF, TXT, DOCX, CSV, XLSX) and add it to the vector store."""
        
        # 1. Determine Loader based on extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        elif ext in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
        try:
            pages = loader.load()
        except Exception as e:
            raise ValueError(f"Error loading file {file_path}: {str(e)}")
        
        # 2. Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(pages)
        
        if not chunks:
             return 0

        # 3. Embed and Store
        vector_store = self.get_vector_store(user_id, workspace_name, embedding_model)
        vector_store.add_documents(documents=chunks)
        
        # 4. Save Metadata
        self.save_file_metadata(user_id, workspace_name, os.path.basename(file_path), len(chunks))
        
        return len(chunks)

    def get_answer(self, user_id: int, workspace_name: str, question: str, llm_model: str, embedding_model: str, k: int = 4) -> Dict[str, Any]:
        """Get an answer from the workspace context."""
        
        vector_store = self.get_vector_store(user_id, workspace_name, embedding_model)
        
        # Retriever
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": k}
        )
        
        # LLM
        llm = ChatOllama(model=llm_model)
        
        # Prompt
        template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke(question)
        source_docs = retriever.invoke(question)
        
        return {
            "answer": response,
            "source_documents": [doc.page_content for doc in source_docs]
        }
