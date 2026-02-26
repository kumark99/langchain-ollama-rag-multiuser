import streamlit as st
import requests
import json
import time

# Backend URL
API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Multi-Workspace", layout="wide")

# --- Session State Initialization ---
if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_workspace" not in st.session_state:
    st.session_state.current_workspace = None

# --- Authentication Helpers ---
def login_user(email, password):
    try:
        response = requests.post(f"{API_URL}/login", json={"email": email, "password": password})
        if response.status_code == 200:
            data = response.json()
            st.session_state.token = True  # Simple boolean for now
            st.session_state.user = data["user"]
            st.success("Login successful!")
            st.rerun()
        else:
            st.error(f"Login failed: {response.json().get('detail')}")
    except Exception as e:
        st.error(f"Connection error: {e}")

def register_user(email, password, name):
    try:
        response = requests.post(f"{API_URL}/register", json={"email": email, "password": password, "name": name})
        if response.status_code == 200:
            st.success("Registration successful! Please login.")
        else:
            st.error(f"Registration failed: {response.json().get('detail')}")
    except Exception as e:
        st.error(f"Connection error: {e}")

def logout():
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.messages = []
    st.session_state.current_workspace = None
    st.rerun()

# --- Data Helpers ---
@st.cache_data(ttl=60)
def get_models():
    try:
        response = requests.get(f"{API_URL}/models")
        return response.json().get("models", []) if response.status_code == 200 else []
    except:
        return []

def get_workspaces(user_id):
    try:
        response = requests.get(f"{API_URL}/workspaces?user_id={user_id}")
        return response.json().get("workspaces", []) if response.status_code == 200 else []
    except:
        return []

def create_workspace(user_id, name):
    try:
        response = requests.post(f"{API_URL}/workspaces", json={"name": name, "user_id": user_id})
        return (True, response.json().get("message")) if response.status_code == 200 else (False, response.json().get("detail", "Unknown error"))
    except Exception as e:
        return False, str(e)

def get_workspace_files(user_id, workspace_name):
    try:
        response = requests.get(f"{API_URL}/workspaces/{workspace_name}/files?user_id={user_id}")
        return response.json().get("files", {}) if response.status_code == 200 else {}
    except:
        return {}


# --- Main App Logic ---
def main_app():
    user_id = st.session_state.user['id']

    # --- Sidebar ---
    with st.sidebar:
        st.header("Workspace Management")
        
        # User Info & Logout
        if st.session_state.user:
            st.info(f"Logged in as: {st.session_state.user['email']}")
            if st.button("Logout"):
                logout()
            st.divider()

        # Workspace Creation
        with st.expander("Create New Workspace"):
            new_ws_name = st.text_input("Workspace Name (alphanumeric)")
            if st.button("Create Workspace"):
                if new_ws_name:
                    success, msg = create_workspace(user_id, new_ws_name)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please enter a name.")

        # Workspace Selection
        try:
            workspaces = get_workspaces(user_id)
        except Exception:
            workspaces = []

        if not workspaces:
            st.info("No workspaces found. Create one above.")
            current_workspace = None
        else:
            index = 0
            if st.session_state.current_workspace in workspaces:
                index = workspaces.index(st.session_state.current_workspace)
            
            current_workspace = st.selectbox("Select Workspace", workspaces, index=index)
            
            if current_workspace != st.session_state.current_workspace:
                st.session_state.current_workspace = current_workspace
                st.session_state.messages = []
                st.rerun()

        st.divider()

        # Refresh capability
        col_header, col_refresh = st.columns([0.7, 0.3])
        with col_header:
            st.write("Model Settings")
        with col_refresh:
            if st.button("🔄", help="Refresh Models"):
                get_models.clear()
                st.rerun()

        models = get_models()
        
        if not models:
            st.warning("No models found. Please ensure Ollama is running.")
            models = ["llama3", "mistral", "nomic-embed-text"]

        llm_indices = [i for i, m in enumerate(models) if "embed" not in m]
        default_llm = llm_indices[0] if llm_indices else 0
        llm_model = st.selectbox("Select LLM Model", models, index=default_llm)
        
        embed_indices = [i for i, m in enumerate(models) if "embed" in m]
        default_embed = embed_indices[0] if embed_indices else (min(len(models)-1, 1) if len(models) > 1 else 0)
        embedding_model = st.selectbox("Select Embedding Model", models, index=default_embed)
        
        st.divider()
        
        # Document Upload
        st.subheader("Document Upload")
        if current_workspace:
            st.markdown(f"**Current Documents:**")
            files_map = get_workspace_files(user_id, current_workspace)
            if files_map:
                for fname, meta in files_map.items():
                    st.caption(f"📄 {fname} ({meta.get('chunk_count','?')} chunks)")
            else:
                st.caption("No documents.")
            
            uploaded_files = st.file_uploader(
                "Upload Documents (PDF, TXT, DOCX, CSV, XLSX)", 
                type=["pdf", "txt", "docx", "doc", "csv", "xlsx", "xls"], 
                accept_multiple_files=True
            )
            
            chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=1000)
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200)
            
            if st.button("Process Documents"):
                if uploaded_files:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_files = len(uploaded_files)
                    success_count = 0
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                        
                        try:
                            # Pass user_id in data for form
                            resp = requests.post(
                                f"{API_URL}/upload", 
                                files=files, 
                                data={
                                    "workspace_name": current_workspace,
                                    "user_id": user_id,
                                    "embedding_model": embedding_model,
                                    "chunk_size": chunk_size,
                                    "chunk_overlap": chunk_overlap
                                }
                            )
                            if resp.status_code == 200:
                                success_count += 1
                            else:
                                st.error(f"Error processing {uploaded_file.name}: {resp.text}")
                        except Exception as e:
                            st.error(f"Failed to connect: {e}")
                        
                        progress_bar.progress((i + 1) / total_files)
                    
                    status_text.text("Finished!")
                    if success_count > 0:
                        st.success(f"Successfully processed {success_count}/{total_files} files!")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.warning("Please upload files first.")
        else:
            st.info("Select a workspace to upload documents.")
                
        st.divider()
        k_value = st.slider("Top K Retrieved", 1, 10, 4)


    # --- Main Content Area ---
    st.title("🤖 Multi-Workspace RAG with Ollama")

    if not current_workspace:
        st.warning("👈 Please create or select a workspace to start chatting.")
        st.stop()

    tab_chat, tab_feedback = st.tabs(["💬 Chat", "📊 Feedback Analytics"])

    with tab_chat:
        st.subheader(f"Workspace: {current_workspace}")

        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant":
                    if "source_documents" in message:
                        with st.expander("View Source Documents"):
                            for j, doc in enumerate(message["source_documents"]):
                                st.markdown(f"**Source {j+1}:**\n{doc}\n---")
                    
                    # Feedback UI
                    def submit_feedback(rating, question, answer, workspace_name):
                        user_email = st.session_state.user['email'] if st.session_state.user else "annoymous"
                        feedback_payload = {
                            "workspace_name": workspace_name,
                            "question": question,
                            "answer": answer,
                            "rating": rating,
                            "user_email": user_email,
                            "user_id": user_id
                        }
                        try:
                            requests.post(f"{API_URL}/rank", json=feedback_payload)
                            st.toast("Feedback submitted! 🌟")
                        except Exception as e:
                            st.error(f"Failed to submit feedback: {e}")

                    feedback_key = f"feedback_{i}_ws_{current_workspace}" 
                    question_text = "Unknown"
                    if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                        question_text = st.session_state.messages[i-1]["content"]

                    rating = st.feedback("stars", key=feedback_key)

                    if f"submitted_{feedback_key}" not in st.session_state:
                        st.session_state[f"submitted_{feedback_key}"] = None
                    
                    if rating is not None and rating != st.session_state[f"submitted_{feedback_key}"]:
                        submit_feedback(rating + 1, question_text, message["content"], current_workspace)
                        st.session_state[f"submitted_{feedback_key}"] = rating

        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    payload = {
                        "workspace_name": current_workspace,
                        "question": prompt,
                        "llm_model": llm_model,
                        "embedding_model": embedding_model,
                        "k": k_value,
                        "user_id": user_id
                    }
                    with st.spinner("Thinking..."):
                        response = requests.post(f"{API_URL}/ask", json=payload)
                        
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("answer", "No answer provided.")
                        source_docs = result.get("source_documents", [])
                        message_placeholder.markdown(answer)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "source_documents": source_docs
                        })
                        st.rerun()
                    else:
                        error_msg = f"Error: {response.text}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"Connection Failed: {e}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    with tab_feedback:
        st.header("Feedback Analytics (My Feedback)")
        if st.button("Refresh Data"):
            st.rerun()
            
        try:
            response = requests.get(f"{API_URL}/feedback?user_id={user_id}")
            if response.status_code == 200:
                feedback_data = response.json().get("feedback", [])
                
                if not feedback_data:
                    st.info("No feedback entries found yet.")
                else:
                    total_feedback = len(feedback_data)
                    avg_rating = sum(item.get('rating', 0) for item in feedback_data) / total_feedback if total_feedback > 0 else 0
                    
                    m1, m2 = st.columns(2)
                    m1.metric("My Reviews", total_feedback)
                    m2.metric("Average Rating", f"{avg_rating:.1f}/5.0")
                    
                    st.divider()
                    st.subheader("Recent Feedback")
                    
                    table_data = []
                    for item in feedback_data:
                        # Safely create preview of answer
                        ans_preview = item.get("answer", "") or ""
                        ans_preview = (ans_preview[:100] + "...") if len(ans_preview) > 100 else ans_preview
                        
                        table_data.append({
                            "Workspace": item.get("workspace", "N/A"),
                            "Rating": "⭐" * int(item.get("rating", 0) or 0),
                            "Question": item.get("question", "N/A"),
                            "Answer": ans_preview
                        })
                    
                    st.dataframe(table_data, use_container_width=True)
            else:
                st.error("Failed to load feedback data.")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")


# --- Switcher ---
if not st.session_state.token:
    st.container()
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h1 style='text-align: center;'>🔐 RAG Knowledge Base</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Secure access for authorized personnel only.</p>", unsafe_allow_html=True)
        
        tab_login, tab_register = st.tabs(["Login", "Register"])
        
        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                if submitted:
                    if email and password:
                        login_user(email, password)
                    else:
                        st.warning("Please fill in all fields.")

        with tab_register:
            with st.form("register_form"):
                new_name = st.text_input("Full Name")
                new_email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Register", use_container_width=True)
                if submitted:
                    if new_email and new_password:
                        register_user(new_email, new_password, new_name)
                    else:
                        st.warning("Please fill in all fields.")

else:
    main_app()
