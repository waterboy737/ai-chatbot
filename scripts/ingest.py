#!/usr/bin/env python3
"""Simple ingestion script: splits a text file into passages, computes embeddings, and stores them in vector store."""
import sys
import uuid
from providers import embeddings
from vector_store import VectorStore

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: ingest.py path/to/text.txt')
        sys.exit(1)
    path = sys.argv[1]
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    vs = VectorStore()
    # naive split: paragraphs
    parts = [p.strip() for p in text.split('\n\n') if p.strip()]
    for p in parts:
        emb = embeddings.get_embedding(p)
        doc_id = str(uuid.uuid4())
        vs.add(doc_id, emb, {'source': path}, p)
        print('added', doc_id)
