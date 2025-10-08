import os
from rag_system import PDFRAGSystem

def print_header():
    print("\n" + "="*80)
    print("PDF RAG SYSTEM - INTERACTIVE DEMO")
    print("="*80)
    print("\nThis system can answer questions based on the following PDFs:")
    print("  - Causal Component Analysis.pdf")
    print("  - Databricks-Big-Book-Of-GenAI-FINAL.pdf")
    print("  - NN on tabular data paper.pdf")
    print("  - Production AI Tutorial.pdf")
    print("="*80 + "\n")

def main():
    print_header()
    
    rag = PDFRAGSystem()
    
    if not os.path.exists("./chroma_db"):
        print("First-time setup: Initializing the RAG system...")
        print("This will process all PDFs and create a vector database.")
        print("This may take a few minutes...\n")
        
        try:
            rag.initialize()
        except ValueError as e:
            print(f"\nError: {e}")
            print("\nPlease make sure you have:")
            print("1. Created a .env file (copy from .env.example)")
            print("2. Added your OpenAI API key to the .env file")
            return
    else:
        print("Loading existing vector database...")
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import Chroma
        import chromadb
        
        try:
            embeddings = OpenAIEmbeddings()
            client = chromadb.PersistentClient(path=rag.persist_directory)
            rag.vectorstore = Chroma(
                client=client,
                embedding_function=embeddings
            )
            rag.setup_qa_chain()
            print("System loaded successfully!\n")
        except Exception as e:
            print(f"\nError loading system: {e}")
            print("\nPlease check your .env file and OPENAI_API_KEY")
            return
    
    print("\nYou can now ask questions! Type 'quit' or 'exit' to end the session.")
    print("Example questions:")
    print("  - What is causal component analysis?")
    print("  - What are the key concepts in GenAI?")
    print("  - How do neural networks perform on tabular data?")
    print("  - What are best practices for production AI?\n")
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the PDF RAG System!")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        print("\nSearching PDFs and generating answer...\n")
        
        try:
            result = rag.ask_question(question)
            
            print("="*80)
            print("ANSWER:")
            print("="*80)
            print(result['result'])
            print("\n" + "="*80)
            print("SOURCES:")
            print("="*80)
            
            for i, doc in enumerate(result['source_documents'], 1):
                source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page_num = doc.metadata.get('page', 'Unknown')
                print(f"\n[{i}] {source_file} (Page {page_num})")
                print(f"    {doc.page_content[:150]}...")
            
            print("\n" + "="*80)
        
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try asking a different question.")

if __name__ == "__main__":
    main()
