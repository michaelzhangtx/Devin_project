# PDF RAG System

A Retrieval Augmented Generation (RAG) system for answering questions based on PDF documents in this repository.

## PDF Files in Repository

This repository contains the following PDF files:
- Causal Component Analysis.pdf (8.3 MB)
- Databricks-Big-Book-Of-GenAI-FINAL.pdf (3.9 MB)
- NN on tabular data paper.pdf (669 KB)
- Production AI Tutorial.pdf (563 KB)

## Setup

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

### Option 1: Interactive Mode (Recommended)

Run the interactive demo for a conversational experience:

```bash
python interactive_demo.py
```

This will:
- Automatically initialize the system on first run
- Allow you to ask multiple questions in a row
- Show answers with source attributions
- Provide a user-friendly interface

### Option 2: Command Line Mode

#### Step 1: Initialize the RAG System

First, you need to process the PDFs and create the vector store. This only needs to be done once:

```bash
python rag_system.py init
```

This will:
- Load all PDF files from the repository
- Extract text from each page
- Split the text into chunks
- Create embeddings using OpenAI
- Store the embeddings in a local ChromaDB database

#### Step 2: Ask Questions

Once initialized, you can ask questions about the content in the PDFs:

```bash
python rag_system.py query "What is causal component analysis?"
```

```bash
python rag_system.py query "What are the key concepts in the Databricks GenAI book?"
```

```bash
python rag_system.py query "How do neural networks perform on tabular data?"
```

The system will:
- Retrieve the most relevant chunks from the PDFs
- Use GPT-3.5-turbo to generate an answer based on the retrieved context
- Show the source documents used to generate the answer

## How It Works

The RAG system uses the following components:

1. **Document Loading**: PyPDFLoader extracts text from PDF files
2. **Text Splitting**: RecursiveCharacterTextSplitter divides text into manageable chunks (1000 chars with 200 char overlap)
3. **Embeddings**: OpenAI's text-embedding-ada-002 model creates vector representations
4. **Vector Store**: ChromaDB stores and retrieves similar document chunks
5. **Question Answering**: LangChain's RetrievalQA chain combines retrieval with GPT-3.5-turbo for answer generation

## System Architecture

```
PDFs → Text Extraction → Chunking → Embeddings → Vector Store (ChromaDB)
                                                          ↓
User Question → Embedding → Similarity Search → Top-k Chunks → LLM → Answer
```

## Examples

### Example 1: Understanding a specific concept
```bash
python rag_system.py query "What is the main contribution of the causal component analysis paper?"
```

### Example 2: Comparing approaches
```bash
python rag_system.py query "What are the advantages and disadvantages of using neural networks on tabular data?"
```

### Example 3: Practical guidance
```bash
python rag_system.py query "What are best practices for production AI according to the tutorial?"
```

## Features

- **Multi-document support**: Processes all PDFs in the repository
- **Source attribution**: Shows which documents and pages were used for answers
- **Efficient retrieval**: Uses vector similarity search for fast context retrieval
- **Persistent storage**: Vector store is saved locally and can be reused
- **Accurate responses**: Uses actual document content rather than making up information

## Troubleshooting

### API Key Issues
If you get an error about OPENAI_API_KEY:
- Make sure you created the .env file
- Verify your API key is correct
- Ensure the .env file is in the same directory as rag_system.py

### No PDFs Found
The system looks for PDF files in the current directory. Make sure you're running the script from the repository root.

### Vector Store Errors
If you need to reinitialize the vector store:
```bash
rm -rf ./chroma_db
python rag_system.py init
```

## Technical Details

- **Embedding Model**: text-embedding-ada-002
- **LLM**: GPT-3.5-turbo
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Top-k Retrieval**: 4 most relevant chunks
- **Vector Store**: ChromaDB (local persistence)

## License

This is a demo RAG system for educational purposes.
