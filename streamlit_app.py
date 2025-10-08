import streamlit as st
import os
from pathlib import Path
from rag_system import PDFRAGSystem
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 PDF RAG System")
st.markdown("### Ask questions about PDF documents using AI")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Available PDF Documents")
    pdf_files = [
        "Causal Component Analysis.pdf (7.9 MB)",
        "Databricks-Big-Book-Of-GenAI-FINAL.pdf (3.8 MB)",
        "NN on tabular data paper.pdf (654 KB)",
        "Production AI Tutorial.pdf (550 KB)"
    ]
    for pdf in pdf_files:
        st.markdown(f"- {pdf}")

with col2:
    st.markdown("#### System Status")
    if os.path.exists("./chroma_db"):
        st.success("✓ System initialized")
    else:
        st.warning("⚠ Needs initialization")

with st.sidebar:
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. **First Time**: The system will automatically initialize when you ask your first question
    2. **Ask Questions**: Type your question in the text box
    3. **View Results**: See the answer with source document citations
    """)
    
    st.markdown("---")
    st.header("💡 Example Questions")
    
    example_questions = [
        "What is causal component analysis?",
        "What are the key concepts in GenAI?",
        "How do neural networks perform on tabular data?",
        "What are best practices for production AI?"
    ]
    
    for i, question in enumerate(example_questions, 1):
        st.markdown(f"**{i}.** {question}")
    
    st.markdown("---")
    st.header("⚙️ Setup")
    st.markdown("""
    **Requirements:**
    - OpenAI API key set in `.env` file
    - Run: `cp .env.example .env`
    - Add your `OPENAI_API_KEY` to `.env`
    """)

st.markdown("---")

def initialize_rag_system():
    try:
        rag = PDFRAGSystem()
        
        if not os.path.exists("./chroma_db"):
            with st.spinner("🔄 First-time setup: Initializing RAG system... This may take a few minutes."):
                rag.initialize()
            st.success("✓ System initialized successfully!")
        else:
            with st.spinner("🔄 Loading existing vector database..."):
                embeddings = OpenAIEmbeddings()
                client = chromadb.PersistentClient(path=rag.persist_directory)
                rag.vectorstore = Chroma(
                    client=client,
                    embedding_function=embeddings
                )
                rag.setup_qa_chain()
            st.success("✓ System loaded successfully!")
        
        return rag
    except ValueError as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("""
        **Please make sure you have:**
        1. Created a `.env` file (copy from `.env.example`)
        2. Added your OpenAI API key to the `.env` file
        """)
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.info("Please check your `.env` file and OPENAI_API_KEY")
        return None

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.initialized = False

question = st.text_input(
    "🔍 Ask a question about the PDFs:",
    placeholder="e.g., What is causal component analysis?",
    key="question_input"
)

if st.button("Get Answer", type="primary") or (question and question != st.session_state.get('last_question', '')):
    if question:
        st.session_state.last_question = question
        
        if not st.session_state.initialized:
            st.session_state.rag_system = initialize_rag_system()
            if st.session_state.rag_system:
                st.session_state.initialized = True
        
        if st.session_state.rag_system:
            try:
                with st.spinner("🤔 Searching PDFs and generating answer..."):
                    result = st.session_state.rag_system.ask_question(question)
                
                st.markdown("---")
                st.markdown("### 💬 Answer")
                st.markdown(result['result'])
                
                st.markdown("---")
                st.markdown("### 📄 Sources")
                
                for i, doc in enumerate(result['source_documents'], 1):
                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    page_num = doc.metadata.get('page', 'Unknown')
                    
                    with st.expander(f"📌 Source {i}: {source_file} (Page {page_num})"):
                        st.text(doc.page_content[:300] + "...")
                
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                st.info("Please try asking a different question.")
    else:
        st.warning("⚠️ Please enter a question.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Powered by LangChain, ChromaDB, and OpenAI GPT-3.5-turbo</p>
</div>
""", unsafe_allow_html=True)
