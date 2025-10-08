import os
from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import chromadb

load_dotenv()

class PDFRAGSystem:
    def __init__(self, pdf_directory: str = ".", persist_directory: str = "./chroma_db"):
        self.pdf_directory = Path(pdf_directory)
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in .env file or environment.")
    
    def load_pdfs(self) -> List:
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.pdf_directory}")
        
        print(f"Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")
        
        documents = []
        for pdf_file in pdf_files:
            print(f"\nLoading {pdf_file.name}...")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            print(f"  Loaded {len(docs)} pages")
            documents.extend(docs)
        
        return documents
    
    def create_vectorstore(self, documents: List):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        print("\nSplitting documents into chunks...")
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} text chunks")
        
        print("\nCreating embeddings and vector store...")
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        print("Vector store created successfully!")
    
    def setup_qa_chain(self):
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        print("QA chain setup complete!")
    
    def ask_question(self, question: str) -> dict:
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call setup_qa_chain first.")
        
        result = self.qa_chain({"query": question})
        return result
    
    def initialize(self):
        documents = self.load_pdfs()
        self.create_vectorstore(documents)
        self.setup_qa_chain()
        print("\n" + "="*80)
        print("RAG System initialized and ready to answer questions!")
        print("="*80 + "\n")

def main():
    import sys
    
    rag = PDFRAGSystem()
    
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        rag.initialize()
        print("\nInitialization complete. You can now use the query mode.")
    elif len(sys.argv) > 1 and sys.argv[1] == "query":
        if len(sys.argv) < 3:
            print("Usage: python rag_system.py query 'Your question here'")
            sys.exit(1)
        
        question = sys.argv[2]
        
        print("Loading existing vector store...")
        embeddings = OpenAIEmbeddings()
        client = chromadb.PersistentClient(path=rag.persist_directory)
        rag.vectorstore = Chroma(
            client=client,
            embedding_function=embeddings
        )
        rag.setup_qa_chain()
        
        print(f"\nQuestion: {question}")
        print("\nGenerating answer...\n")
        
        result = rag.ask_question(question)
        
        print("="*80)
        print("ANSWER:")
        print("="*80)
        print(result['result'])
        print("\n" + "="*80)
        print("SOURCE DOCUMENTS:")
        print("="*80)
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"\n[Source {i}]")
            print(f"File: {doc.metadata.get('source', 'Unknown')}")
            print(f"Page: {doc.metadata.get('page', 'Unknown')}")
            print(f"Content preview: {doc.page_content[:200]}...")
    else:
        print("RAG System - PDF Question Answering")
        print("\nUsage:")
        print("  Initialize the system (first time):")
        print("    python rag_system.py init")
        print("\n  Query the system:")
        print("    python rag_system.py query 'Your question here'")

if __name__ == "__main__":
    main()
