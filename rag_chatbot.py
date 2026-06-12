

import os
import sys
import textwrap
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# ──────────────────────────────────────────────
# CONFIG — edit these to suit your setup
# ──────────────────────────────────────────────
COLLECTION_NAME = "rag_docs"
CHUNK_SIZE      = 300        # characters per chunk
CHUNK_OVERLAP   = 50         # overlap between chunks
TOP_K           = 3          # number of chunks to retrieve

# LLM backend selection
# Options: "ollama" | "openai" | "groq"
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL    = "llama3"

OPENAI_MODEL    = "gpt-3.5-turbo"
GROQ_BASE_URL   = "https://api.groq.com/openai/v1"
GROQ_MODEL      = "llama3-8b-8192"

# ──────────────────────────────────────────────
# SAMPLE DOCUMENTS (replace or extend these)
# ──────────────────────────────────────────────
SAMPLE_DOCS = [
    {
        "id": "doc1",
        "text": """
        ChromaDB is an open-source embedding database designed for AI applications.
        It runs embedded inside your application process and supports both in-memory
        and persistent storage modes. ChromaDB uses SQLite for metadata and HNSWLib
        for fast approximate nearest neighbour (ANN) vector search. It stores vectors,
        raw documents, and metadata together in one local package, making it ideal for
        prototyping RAG systems without any cloud dependencies.
        """
    },
    {
        "id": "doc2",
        "text": """
        Pinecone is a fully managed, cloud-native vector database offered as a SaaS
        product. It abstracts all infrastructure concerns away from the developer.
        Data is hosted across AWS, GCP, or Azure depending on the chosen region.
        Pinecone uses a proprietary closed-source architecture that shards data across
        multiple nodes for high availability. It is best suited for production workloads
        that require scalability and low-latency similarity search at massive scale.
        """
    },
    {
        "id": "doc3",
        "text": """
        FAISS (Facebook AI Similarity Search) is a highly optimised C++ library for
        efficient similarity search and clustering of dense vectors. Unlike ChromaDB
        or Pinecone, FAISS is not a database — it is a mathematical library that
        operates exclusively on vectors in RAM or GPU memory. It does not store text
        or metadata; you must maintain a separate mapping from vector IDs back to your
        documents. FAISS is the fastest option for offline, batch-scale vector search.
        """
    },
    {
        "id": "doc4",
        "text": """
        Retrieval-Augmented Generation (RAG) is an AI framework that enhances large
        language model (LLM) responses by supplying relevant external context at
        inference time. The pipeline works as follows: documents are split into chunks
        and converted to vector embeddings, which are stored in a vector database.
        When a user submits a query, it is also embedded and used to retrieve the
        top-k most similar chunks. Those chunks are injected into the LLM prompt as
        context, grounding the model's response in factual, domain-specific information
        rather than relying solely on parametric memory.
        """
    },
    {
        "id": "doc5",
        "text": """
        Sentence Transformers is a Python library that provides pre-trained models for
        generating dense vector embeddings from text. Models like all-MiniLM-L6-v2
        produce 384-dimensional embeddings and run entirely on CPU, making them
        practical for local RAG systems. These embeddings capture semantic meaning,
        allowing similarity search to surface conceptually related content even when
        the exact keywords differ between the query and the stored documents.
        """
    },
]

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def build_vector_store(docs: list[dict]) -> chromadb.Collection:
    """Chunk documents and load them into a ChromaDB collection."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.Client()  # in-memory; use PersistentClient("./chroma_db") to persist

    # Delete collection if it already exists (fresh start each run)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks, all_ids, all_meta = [], [], []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc['id']}_chunk{i}")
            all_meta.append({"source": doc["id"]})

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_meta)
    print(f"  ✓ Indexed {len(all_chunks)} chunks from {len(docs)} documents.\n")
    return collection


def retrieve(collection: chromadb.Collection, query: str) -> list[str]:
    """Return the top-k most relevant chunks for a query."""
    results = collection.query(query_texts=[query], n_results=TOP_K)
    return results["documents"][0]  # list of chunk strings


def build_prompt(context_chunks: list[str], query: str) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so honestly.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""


def get_llm_client() -> tuple[OpenAI, str]:
    """Return an OpenAI-compatible client and the model name based on LLM_BACKEND."""
    backend = LLM_BACKEND.lower()

    if backend == "ollama":
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        return client, OLLAMA_MODEL

    elif backend == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            sys.exit("Error: GROQ_API_KEY environment variable not set.")
        client = OpenAI(base_url=GROQ_BASE_URL, api_key=api_key)
        return client, GROQ_MODEL

    else:  # openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            sys.exit("Error: OPENAI_API_KEY environment variable not set.")
        client = OpenAI(api_key=api_key)
        return client, OPENAI_MODEL


def ask(client: OpenAI, model: str, prompt: str) -> str:
    """Send the prompt to the LLM and return the response text."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def load_txt_files(directory: str) -> list[dict]:
    """
    Optional helper: load all .txt files from a folder as additional documents.
    Usage: pass the returned list into build_vector_store() alongside SAMPLE_DOCS.
    """
    docs = []
    if not os.path.isdir(directory):
        return docs
    for fname in os.listdir(directory):
        if fname.endswith(".txt"):
            path = os.path.join(directory, fname)
            with open(path, "r", encoding="utf-8") as f:
                docs.append({"id": fname, "text": f.read()})
    return docs


# ──────────────────────────────────────────────
# MAIN CHAT LOOP
# ──────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("   RAG Mini Chatbot  |  Type 'quit' to exit")
    print("="*55)
    print(f"\nBackend : {LLM_BACKEND.upper()} | Embedding: all-MiniLM-L6-v2")
    print("\nIndexing documents...")

    # To load your own .txt files, uncomment:
    # extra_docs = load_txt_files("./my_docs")
    # all_docs = SAMPLE_DOCS + extra_docs
    all_docs = SAMPLE_DOCS

    collection = build_vector_store(all_docs)
    client, model = get_llm_client()

    print("Chatbot ready. Ask me anything about the indexed documents.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break

        # Retrieve + generate
        chunks = retrieve(collection, query)
        prompt = build_prompt(chunks, query)
        answer = ask(client, model, prompt)

        # Pretty-print
        print(f"\nBot: {textwrap.fill(answer, width=70)}\n")

        # Optional: show retrieved sources
        sources = collection.query(query_texts=[query], n_results=TOP_K)["metadatas"][0]
        source_ids = list(dict.fromkeys(s["source"] for s in sources))
        print(f"  [Sources used: {', '.join(source_ids)}]\n")


if __name__ == "__main__":
    main()
