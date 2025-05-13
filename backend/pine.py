#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv

import openai
from pinecone import Pinecone

# ─── LOAD .env ───────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")  # ← replace with your actual Pinecone region if different

# Check if keys are available
if not PINECONE_API_KEY:
    print("WARNING: PINECONE_API_KEY is not set. Interview search functionality will not work.")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set. API calls to OpenAI will fail.")

openai.api_key = OPENAI_API_KEY

# ─── INITIALIZE PINECONE ────────────────────────────────────────────
try:
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    index = pc.Index(
        name="cq-transcripts-1",
        pool_threads=50,
        connection_pool_maxsize=50,
    )
    print("Pinecone initialized successfully.")
except Exception as e:
    print(f"ERROR initializing Pinecone: {str(e)}")
    index = None

# ─── QUERY + STREAM FUNCTION ────────────────────────────────────────
def query_and_stream(query: str):
    if not index:
        print("ERROR: Pinecone index not initialized")
        return
        
    # 1) Embed the query
    emb_resp = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    vector = emb_resp.data[0].embedding

    # 2) Query Pinecone in 'innabox' namespace
    pc_resp = index.query(
        vector=vector,
        top_k=3,
        include_metadata=True,
        namespace="innabox"
    )

    # 3) Stream back each match (with debug)
    for match in pc_resp.matches:
        meta = match.metadata or {}
        # DEBUG: see exactly what's in metadata
        print("DEBUG metadata:", meta)

        # pick the first available snippet field
        snippet = None
        for field in ("text", "snippet", "content", "transcript", "body"):
            if field in meta:
                snippet = meta[field]
                break
        if snippet is None:
            snippet = "<no snippet available>"

        # pick a filename (or fallback to 'source')
        filename = meta.get("filename", meta.get("source", "unknown"))

        print(f">>> [{filename}]\n{snippet}\n")

# ─── MAIN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Query: ").strip()

    print(f"\n🔍 Searching for: \"{user_query}\"\n")
    query_and_stream(user_query)
