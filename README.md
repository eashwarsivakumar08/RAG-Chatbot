

# RAG Mini Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot using claude

## What it does
- Loads documents into a ChromaDB vector store
- Embeds queries using Sentence Transformers (all-MiniLM-L6-v2)
- Retrieves the top-3 most relevant chunks via cosine similarity
- Sends retrieved context + query to an LLM for a grounded response

## Tech Stack
- **Vector DB:** ChromaDB
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **LLM:** Ollama (Llama 3) / Groq / OpenAI (configurable)

## Setup
```bash
pip install -r requirements.txt
