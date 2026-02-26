from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Depends, Query
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
import shutil
import os
import subprocess
import json
import sqlite3
import time
from passlib.context import CryptContext
from email_validator import validate_email, EmailNotValidError
from .rag import RAGService

# --- Database & Auth Setup ---
DB_FILE = "data/users.db"
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def get_db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT
        )
    ''')
    conn.commit()
    conn.close()

if not os.path.exists("data"):
    os.makedirs("data")
init_db()

# --- Application Setup ---
app = FastAPI(title="LangChain RAG Multi-Workspace with Users")
rag_service = RAGService()

# --- Models ---
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserProfile(BaseModel):
    id: int
    email: str
    name: Optional[str]

class WorkspaceCreateRequest(BaseModel):
    name: str
    user_id: int

class AskRequest(BaseModel):
    question: str
    workspace_name: str
    user_id: int
    llm_model: str = "llama3"
    embedding_model: str = "nomic-embed-text"
    k: int = 4

class RankRequest(BaseModel):
    question: str
    answer: str
    rating: int
    user_id: int
    workspace_name: str
    comment: Optional[str] = None
    user_email: Optional[str] = None

# --- Auth Helpers ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# --- Endpoints: Auth ---
@app.post("/register")
async def register(user: UserRegister):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        hashed_pw = get_password_hash(user.password)
        c.execute("INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)", 
                  (user.email, hashed_pw, user.name))
        conn.commit()
        return {"message": "User registered successfully", "email": user.email}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")
    finally:
        conn.close()

@app.post("/login")
async def login(user: UserLogin):
    conn = get_db_connection()
    c = conn.cursor()
    user_row = c.execute("SELECT * FROM users WHERE email = ?", (user.email,)).fetchone()
    conn.close()
    
    if not user_row or not verify_password(user.password, user_row['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    return {
        "message": "Login successful",
        "user": {
            "id": user_row['id'], 
            "email": user_row['email'],
            "name": user_row['name']
        }
    }

# --- Endpoints: Core ---
@app.get("/models")
async def list_models():
    """List available Ollama models"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
             return {"models": []}
        lines = result.stdout.strip().split('\n')[1:]
        models = [line.split()[0] for line in lines if line.strip()]
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}

@app.get("/workspaces")
async def get_workspaces(user_id: int):
    try:
        workspaces = rag_service.list_workspaces(user_id)
        return {"workspaces": workspaces}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workspaces")
async def create_workspace(request: WorkspaceCreateRequest):
    try:
        # Validate workspace name is alphanumeric
        if not request.name.isalnum():
             raise HTTPException(status_code=400, detail="Workspace name must be alphanumeric.")
             
        rag_service.create_workspace(request.user_id, request.name)
        return {"message": f"Workspace '{request.name}' created.", "name": request.name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/workspaces/{workspace_name}")
async def delete_workspace(workspace_name: str, user_id: int):
    try:
        success = rag_service.delete_workspace(user_id, workspace_name)
        if not success:
            raise HTTPException(status_code=404, detail="Workspace not found")
        return {"message": f"Workspace '{workspace_name}' deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workspaces/{workspace_name}/files")
async def get_workspace_files(workspace_name: str, user_id: int):
    try:
        files = rag_service.get_workspace_files(user_id, workspace_name)
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    workspace_name: str = Form(...),
    user_id: int = Form(...),
    embedding_model: str = Form("nomic-embed-text"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    try:
        workspace_path = rag_service.get_workspace_path(user_id, workspace_name)
        if not os.path.exists(workspace_path):
             raise HTTPException(status_code=404, detail=f"Workspace '{workspace_name}' not found for user.")

        upload_dir = os.path.join(workspace_path, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        num_chunks = rag_service.process_file(user_id, workspace_name, file_path, embedding_model, chunk_size, chunk_overlap)
        
        return {
            "filename": file.filename, 
            "workspace": workspace_name,
            "chunks": num_chunks, 
            "message": "File processed and added to workspace."
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: AskRequest):
    try:
        workspace_path = rag_service.get_workspace_path(request.user_id, request.workspace_name)
        if not os.path.exists(workspace_path):
             raise HTTPException(status_code=404, detail=f"Workspace '{request.workspace_name}' not found.")
        
        chroma_path = rag_service.get_chroma_path(request.user_id, request.workspace_name)
        if not os.path.exists(chroma_path) or not os.listdir(chroma_path):
             raise HTTPException(status_code=400, detail="No documents indexed in this workspace.")

        result = rag_service.get_answer(
            request.user_id,
            request.workspace_name, 
            request.question, 
            request.llm_model, 
            request.embedding_model, 
            request.k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rank")
async def rank_answer(request: RankRequest):
    log_entry = {
        "timestamp": time.time(),
        "user_id": request.user_id,
        "user_email": request.user_email,
        "workspace": request.workspace_name,
        "question": request.question,
        "answer": request.answer,
        "rating": request.rating,
        "comment": request.comment
    }
    
    log_file = "data/feedback_log.json"
    os.makedirs("data", exist_ok=True)
    
    existing_data = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                pass
    
    existing_data.append(log_entry)
    
    with open(log_file, "w") as f:
        json.dump(existing_data, f, indent=4)
        
    return {"message": "Feedback received"}

@app.get("/feedback")
async def get_feedback(user_id: int, limit: int = 50):
    log_file = "data/feedback_log.json"
    if not os.path.exists(log_file):
        return {"feedback": []}
    try:
        with open(log_file, "r") as f:
            data = json.load(f)
            # Filter by user_id
            user_feedback = [d for d in data if d.get('user_id') == user_id]
            return {"feedback": user_feedback[::-1][:limit]}
    except:
        return {"feedback": []}
