import streamlit as st
import os
import json
import re
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse
from datetime import datetime
import time

# LangChain components
from langchain.document_loaders import JSONLoader
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Custom CSS for modern design
def load_css():
    st.markdown("""
    <style>
    :root {
        --primary: #6d28d9;
        --primary-dark: #4c1d95;
        --secondary: #f5f3ff;
        --text: #1e293b;
        --text-light: #64748b;
        --bg: #ffffff;
        --card-bg: #f8fafc;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --radius: 12px;
    }
    
    [data-theme="dark"] {
        --primary: #8b5cf6;
        --primary-dark: #6d28d9;
        --secondary: #1e1b4b;
        --text: #e2e8f0;
        --text-light: #94a3b8;
        --bg: #0f172a;
        --card-bg: #1e293b;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
    }
    
    html {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: var(--bg);
        color: var(--text);
        transition: all 0.3s ease;
    }
    
    .header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--primary), #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subheader {
        font-size: 1.1rem;
        color: var(--text-light);
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: var(--card-bg);
        border-radius: var(--radius);
        padding: 1.5rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: var(--primary);
    }
    
    .btn {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: var(--radius);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        width: 100%;
        text-align: center;
        margin: 0.25rem 0;
    }
    
    .btn:hover {
        background-color: var(--primary-dark);
        transform: translateY(-1px);
    }
    
    .btn-secondary {
        background-color: var(--card-bg);
        color: var(--primary);
        border: 1px solid var(--primary);
    }
    
    .btn-secondary:hover {
        background-color: var(--primary);
        color: white;
    }
    
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .chat-bubble {
        max-width: 80%;
        padding: 1rem 1.25rem;
        border-radius: var(--radius);
        position: relative;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-bubble {
        align-self: flex-end;
        background-color: var(--primary);
        color: white;
        border-bottom-right-radius: 0;
    }
    
    .assistant-bubble {
        align-self: flex-start;
        background-color: var(--card-bg);
        color: var(--text);
        border-bottom-left-radius: 0;
        box-shadow: var(--shadow);
    }
    
    .summary-bubble {
        align-self: flex-start;
        background-color: var(--secondary);
        color: var(--text);
        border-left: 4px solid var(--primary);
    }
    
    .timestamp {
        font-size: 0.75rem;
        color: var(--text-light);
        margin-top: 0.5rem;
        text-align: right;
    }
    
    .file-uploader {
        border: 2px dashed var(--primary);
        border-radius: var(--radius);
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }
    
    .file-uploader:hover {
        background-color: var(--secondary);
    }
    
    .spinner {
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Toggle switch for theme */
    .theme-switch {
        position: fixed;
        bottom: 1rem;
        right: 1rem;
        z-index: 999;
    }
    
    .theme-switch label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        cursor: pointer;
        color: var(--text);
    }
    
    .switch {
        position: relative;
        display: inline-block;
        width: 50px;
        height: 24px;
    }
    
    .switch input {
        opacity: 0;
        width: 0;
        height: 0;
    }
    
    .slider {
        position: absolute;
        cursor: pointer;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: #ccc;
        transition: .4s;
        border-radius: 34px;
    }
    
    .slider:before {
        position: absolute;
        content: "";
        height: 16px;
        width: 16px;
        left: 4px;
        bottom: 4px;
        background-color: white;
        transition: .4s;
        border-radius: 50%;
    }
    
    input:checked + .slider {
        background-color: var(--primary);
    }
    
    input:checked + .slider:before {
        transform: translateX(26px);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header {
            font-size: 2rem;
        }
        
        .card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

class PDFProcessor:
    """Handles PDF parsing with LlamaParse and saves to Markdown"""
    
    def __init__(self, output_dir: str = "markdown_output"):
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_PARSE_KEY"),
            result_type="markdown",
            verbose=True,
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def parse_pdf_to_markdown(self, pdf_path: str) -> Optional[str]:
        """Process PDF and save as markdown file"""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            with st.spinner("Processing PDF..."):
                documents = self.parser.load_data(pdf_path)
                markdown_text = "\n".join(doc.text for doc in documents)
                
                # Save to markdown file
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                output_path = os.path.join(self.output_dir, f"{base_name}.md")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_text)
                    
                return output_path
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None

class RAGSystem:
    """RAG System that works with saved Markdown files"""
    
    def __init__(self):
        # Initialize components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", " "]
        )
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize storage
        self.vectorstore = None
        self.docstore = InMemoryStore()
        self.retriever = None
        
    def load_markdown_file(self, md_path: str, metadata: dict = {}) -> List[Document]:
        """Load and split a markdown file into documents"""
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
            
            doc = Document(
                page_content=markdown_text,
                metadata={"source": md_path, **metadata}
            )
            return self.text_splitter.split_documents([doc])
        except Exception as e:
            st.error(f"Error loading markdown file: {e}")
            return []
    
    def initialize_retriever(self, documents: List[Document]):
        """Initialize the retriever with documents"""
        self.vectorstore = Chroma.from_documents(documents, self.embedding_model)
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.text_splitter
        )
        self.retriever.add_documents(documents)
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant document chunks for a query"""
        try:
            if not self.retriever:
                raise ValueError("Retriever not initialized. Please add documents first.")
                
            results = self.retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in results[:top_k]])
            return context
        except Exception as e:
            return f"Error during retrieval: {str(e)}"
    
    def json_to_obj(self, json_str: str) -> dict:
        """Clean and parse JSON output from LLM"""
        cleaned = re.sub(r"^```json\s*|\s*```$", "", json_str.strip(), flags=re.IGNORECASE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON: {e}")
            st.error(f"Raw content was: {json_str}")
            return {"answer": json_str}

class PDFQASystem:
    """System that uses saved Markdown files for QA and summarization"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.rag_system = RAGSystem()
        self.loaded_documents = []
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        self.chat_history = []
        self.is_retriever_initialized = False
    
    def process_pdf_and_save_markdown(self, pdf_path: str) -> Optional[str]:
        """Process a PDF file and save as markdown"""
        return self.pdf_processor.parse_pdf_to_markdown(pdf_path)
    
    def load_markdown_file(self, md_path: str, metadata: dict = {}) -> bool:
        """Load a markdown file into the system"""
        try:
            documents = self.rag_system.load_markdown_file(md_path, metadata)
            if not documents:
                return False
                
            self.loaded_documents.extend(documents)
            
            if not self.is_retriever_initialized:
                self.rag_system.initialize_retriever(documents)
                self.is_retriever_initialized = True
            else:
                self.rag_system.retriever.add_documents(documents)
                
            return True
        except Exception as e:
            st.error(f"Error loading markdown file: {e}")
            return False
    
    def ask_question(self, question: str) -> Dict[str, str]:
        """Your original QA prompt"""
        if not self.loaded_documents:
            return {"question": question, "answer": "No documents loaded. Please load markdown files first."}
            
        QA_PROMPT = """  
        You are a helpful assistant that answers questions based strictly on the provided context.
        Answer the question clearly and concisely using only the information from the context.
        If the context doesn't contain the answer, say "I don't know."

        Context:
        {context}

        Question: {question}

        Please respond in JSON format with 'question' and 'answer' fields.
        """
        
        context = self.rag_system.retrieve_relevant_documents(question)
        prompt = QA_PROMPT.format(context=context, question=question)
        
        with st.spinner("Thinking..."):
            response = self.llm.invoke(prompt)
            result = self.rag_system.json_to_obj(response.content)
            
            # Save to chat history (new messages at the beginning)
            self.chat_history.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "type": "answer",
                "content": result["answer"]
            })
            self.chat_history.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "type": "question",
                "content": question
            })
            
            self.save_chat_history()
            return result
    
    def generate_concise_summary(self) -> Dict[str, str]:
        """Your original summary prompt"""
        if not self.loaded_documents:
            return {"error": "No documents loaded for summarization"}
            
        full_text = "\n\n".join([doc.page_content for doc in self.loaded_documents])
        
        SUMMARY_PROMPT = """
        Task:
Analyze the provided PDF document and generate a structured summary that captures key information clearly and concisely. Adapt the output format based on the document type (technical, legal, course material, article, etc.).

Output Guidelines:
Title & Main Topic:

Identify the document's primary subject (e.g., "Microcontrollers," "Legal Contract Terms").

Summarize the core purpose/theme in 1–2 sentences.

Key Sections/Components:

Break down the document into logical subtopics (headings, chapters, or themes).

For each subtopic, list bullet points (1–2 lines each) covering:

Definitions (if technical).

Critical features/functions (for technical docs).

Key arguments/findings (for articles/research).

Clauses/obligations (for legal docs).

Applications/Examples (if applicable):

Highlight real-world uses, case studies, or scenarios.

Comparative Analysis (if relevant):

Contrast concepts (e.g., "Interrupts vs. Polling," "Analog vs. Digital Signals").

Technical/Legal Nuances:

For technical docs: Note specs (e.g., voltage ranges, memory types).

For legal docs: Summarize obligations, restrictions, penalties.


Formatting Rules:
Use bold headers for main topics (###).

Bullet points for brevity (-).

Italics for emphasis or definitions (*term*).

Tables/figures only if referenced in the text.

Examples of Adaptability:

Technical PDF (e.g., Microcontrollers): Focus on components, specs, applications.

Course/Lesson PDF: Summarize modules, key learnings, exercises.

Legal PDF: Extract clauses, parties, deadlines, penalties.

Article/Research PDF: Highlight thesis, methodology, conclusions.
        {context}
        """
        
        prompt = SUMMARY_PROMPT.format(context=full_text)
        
        with st.spinner("Generating summary..."):
            response = self.llm.invoke(prompt)
            
            # Save to chat history (new summary at the beginning)
            self.chat_history.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "type": "summary",
                "content": response.content
            })
            self.save_chat_history()
            
            return {"summary": response.content}
    
    def save_chat_history(self):
        """Save chat history to a JSON file"""
        try:
            with open("chat_history.json", "w", encoding="utf-8") as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Error saving chat history: {e}")
    
    def load_chat_history(self):
        """Load chat history from JSON file"""
        try:
            if os.path.exists("chat_history.json"):
                with open("chat_history.json", "r", encoding="utf-8") as f:
                    self.chat_history = json.load(f)
        except Exception as e:
            st.error(f"Error loading chat history: {e}")

# Initialize the system and load CSS
if "pdf_qa_system" not in st.session_state:
    st.session_state.pdf_qa_system = PDFQASystem()
    st.session_state.pdf_qa_system.load_chat_history()
    st.session_state.file_processed = False

load_css()

# Theme toggle
def toggle_theme():
    if st.session_state.get("theme", "light") == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"

# Streamlit UI
st.markdown('<h1 class="header">PDF Intelligence Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload PDFs, ask questions, and get AI-powered insights</p>', unsafe_allow_html=True)

# File upload section
with st.container():
    st.markdown('<div class="card"><div class="card-title">Upload Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None and not st.session_state.file_processed:
        # Save the uploaded file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the PDF
        with st.spinner(f"Processing {uploaded_file.name}..."):
            md_path = st.session_state.pdf_qa_system.process_pdf_and_save_markdown(temp_pdf_path)
            
            if md_path:
                success = st.session_state.pdf_qa_system.load_markdown_file(
                    md_path, 
                    {"title": uploaded_file.name}
                )
                
                if success:
                    st.session_state.file_processed = True
                    st.success(f"✅ Document loaded successfully: {uploaded_file.name}")
                else:
                    st.error("Failed to load the PDF content")
            
            # Clean up temp file
            try:
                os.remove(temp_pdf_path)
            except:
                pass

# Main interaction area
if st.session_state.pdf_qa_system.loaded_documents:
    with st.container():
        st.markdown('<div class="card"><div class="card-title">Document Actions</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Ask a Question", key="ask_btn", help="Ask questions about the document"):
                st.session_state.current_mode = "qa"
        with col2:
            if st.button("Generate Summary", key="summary_btn", help="Get a comprehensive summary"):
                st.session_state.current_mode = "summary"
                summary = st.session_state.pdf_qa_system.generate_concise_summary()
                
                with st.container():
                    st.markdown('<div class="card"><div class="card-title">Document Summary</div>', unsafe_allow_html=True)
                    st.markdown(summary["summary"])
        
        if "current_mode" in st.session_state and st.session_state.current_mode == "qa":
            with st.container():
                st.markdown('<div class="card"><div class="card-title">Ask a Question</div>', unsafe_allow_html=True)
                question = st.text_input("Type your question here:", key="question_input", label_visibility="collapsed")
                
                if question:
                    answer = st.session_state.pdf_qa_system.ask_question(question)
                    
                    # Display the answer in a chat bubble
                    st.markdown(f"""
                    <div class="chat-container">
                        <div class="chat-bubble user-bubble">
                            {question}
                            <div class="timestamp">{datetime.now().strftime('%H:%M')}</div>
                        </div>
                        <div class="chat-bubble assistant-bubble">
                            {answer['answer']}
                            <div class="timestamp">{datetime.now().strftime('%H:%M')}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Chat history display (showing most recent first)
if st.session_state.pdf_qa_system.chat_history:
    with st.container():
        st.markdown('<div class="card"><div class="card-title">Conversation History</div>', unsafe_allow_html=True)
        
        for item in st.session_state.pdf_qa_system.chat_history:
            timestamp = datetime.fromisoformat(item["timestamp"]).strftime('%H:%M')
            
            if item["type"] == "question":
                st.markdown(f"""
                <div class="chat-container">
                    <div class="chat-bubble user-bubble">
                        {item["content"]}
                        <div class="timestamp">{timestamp}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif item["type"] == "answer":
                st.markdown(f"""
                <div class="chat-container">
                    <div class="chat-bubble assistant-bubble">
                        {item["content"]}
                        <div class="timestamp">{timestamp}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif item["type"] == "summary":
                st.markdown(f"""
                <div class="chat-container">
                    <div class="chat-bubble summary-bubble">
                        <strong>Document Summary:</strong>
                        {item["content"]}
                        <div class="timestamp">{timestamp}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Theme toggle switch
st.markdown("""
<div class="theme-switch">
    <label>
        <span>🌙</span>
        <div class="switch">
            <input type="checkbox" onchange="toggleTheme()">
            <span class="slider"></span>
        </div>
        <span>☀️</span>
    </label>
</div>

<script>
function toggleTheme() {
    const html = document.querySelector('html');
    const currentTheme = html.getAttribute('data-theme') || 'light';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    html.setAttribute('data-theme', newTheme);
    
    // Store preference in session
    fetch('/_stcore/host-config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ theme: newTheme })
    });
}
</script>
""", unsafe_allow_html=True)

# Initialize theme
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('streamlit-theme') || 'light';
    document.querySelector('html').setAttribute('data-theme', savedTheme);
    if (savedTheme === 'dark') {
        document.querySelector('.theme-switch input').checked = true;
    }
});
</script>
""", unsafe_allow_html=True)