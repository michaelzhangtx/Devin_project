import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def test_pdf_loading():
    print("="*80)
    print("Testing PDF Loading Functionality")
    print("="*80)
    
    pdf_directory = Path(".")
    pdf_files = list(pdf_directory.glob("*.pdf"))
    
    if not pdf_files:
        print("ERROR: No PDF files found in current directory!")
        return False
    
    print(f"\nFound {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name} ({pdf.stat().st_size / 1024 / 1024:.2f} MB)")
    
    all_documents = []
    total_pages = 0
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            num_pages = len(docs)
            total_pages += num_pages
            all_documents.extend(docs)
            
            print(f"  ✓ Successfully loaded {num_pages} pages")
            
            if num_pages > 0:
                first_page = docs[0]
                preview = first_page.page_content[:200].replace('\n', ' ')
                print(f"  ✓ First page preview: {preview}...")
                print(f"  ✓ Metadata: {first_page.metadata}")
            
        except Exception as e:
            print(f"  ✗ ERROR loading {pdf_file.name}: {e}")
            return False
    
    print("\n" + "="*80)
    print("PDF Loading Test Results")
    print("="*80)
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Total pages extracted: {total_pages}")
    print(f"Total documents loaded: {len(all_documents)}")
    
    if len(all_documents) > 0:
        total_chars = sum(len(doc.page_content) for doc in all_documents)
        avg_chars = total_chars / len(all_documents)
        print(f"Total characters: {total_chars:,}")
        print(f"Average characters per page: {avg_chars:.0f}")
        print("\n✓ All PDF loading tests PASSED!")
        return True
    else:
        print("\n✗ No documents were loaded!")
        return False

def test_text_splitting():
    print("\n" + "="*80)
    print("Testing Text Splitting Functionality")
    print("="*80)
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    
    pdf_files = list(Path(".").glob("*.pdf"))
    if not pdf_files:
        print("ERROR: No PDF files found!")
        return False
    
    loader = PyPDFLoader(str(pdf_files[0]))
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    splits = text_splitter.split_documents(docs)
    
    print(f"\nTesting with: {pdf_files[0].name}")
    print(f"Original documents (pages): {len(docs)}")
    print(f"Text chunks created: {len(splits)}")
    
    if len(splits) > 0:
        print(f"\nSample chunk preview:")
        print(f"  Length: {len(splits[0].page_content)} characters")
        print(f"  Content: {splits[0].page_content[:150]}...")
        print(f"  Metadata: {splits[0].metadata}")
        print("\n✓ Text splitting test PASSED!")
        return True
    else:
        print("\n✗ Text splitting test FAILED!")
        return False

def test_imports():
    print("="*80)
    print("Testing Required Imports")
    print("="*80)
    
    imports_to_test = [
        ("langchain", "LangChain core"),
        ("langchain_community.document_loaders", "PDF Loader"),
        ("langchain_community.vectorstores", "Vector Store"),
        ("langchain_openai", "OpenAI Integration"),
        ("langchain.text_splitter", "Text Splitter"),
        ("langchain.chains", "Chains"),
        ("pypdf", "PyPDF"),
        ("chromadb", "ChromaDB"),
    ]
    
    all_passed = True
    for module_name, description in imports_to_test:
        try:
            __import__(module_name)
            print(f"  ✓ {description:30} - {module_name}")
        except ImportError as e:
            print(f"  ✗ {description:30} - {module_name} - ERROR: {e}")
            all_passed = False
    
    if all_passed:
        print("\n✓ All import tests PASSED!")
    else:
        print("\n✗ Some imports FAILED!")
    
    return all_passed

def main():
    print("\n" + "="*80)
    print("RAG SYSTEM VERIFICATION TEST SUITE")
    print("="*80)
    print("\nThis test suite verifies the RAG system components that")
    print("don't require an OpenAI API key.")
    print("="*80 + "\n")
    
    tests_passed = []
    
    tests_passed.append(("Imports", test_imports()))
    tests_passed.append(("PDF Loading", test_pdf_loading()))
    tests_passed.append(("Text Splitting", test_text_splitting()))
    
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in tests_passed:
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"{test_name:20} - {status}")
    
    all_passed = all(passed for _, passed in tests_passed)
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nThe RAG system is ready to use with an OpenAI API key.")
        print("To fully test the system:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your OPENAI_API_KEY to the .env file")
        print("  3. Run: python rag_system.py init")
        print("  4. Run: python rag_system.py query 'Your question'")
        print("  Or use: python interactive_demo.py for interactive mode")
    else:
        print("✗ SOME TESTS FAILED!")
        print("Please check the errors above.")
    
    print("="*80 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
