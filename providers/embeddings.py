import os
import time
import json
from typing import List

# Provider selection comes from env var list
PROVIDER_ORDER = [p.strip() for p in os.getenv('EMBEDDINGS_PROVIDERS', 'huggingface,cohere,openai').split(',') if p.strip()]

# Import provider-specific modules
import requests
import numpy as np


def _call_huggingface(text: str):
    """Call Hugging Face embeddings endpoint (Inference API)."""
    hf_key = os.getenv('HUGGINGFACE_API_KEY')
    if not hf_key:
        raise RuntimeError('HUGGINGFACE_API_KEY not set')
    url = 'https://api-inference.huggingface.co/embeddings'
    model = os.getenv('HF_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    headers = { 'Authorization': f'Bearer {hf_key}' }
    payload = { 'model': model, 'input': text }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Expecting {'embedding': [...] } or similar
    if isinstance(data, dict) and 'embedding' in data:
        return np.array(data['embedding'], dtype='float32')
    # Some HF endpoints return {'data':[{'embedding':...}]}
    if isinstance(data, dict) and 'data' in data and len(data['data'])>0 and 'embedding' in data['data'][0]:
        return np.array(data['data'][0]['embedding'], dtype='float32')
    raise RuntimeError('Unexpected HF embeddings response')


def _call_cohere(text: str):
    key = os.getenv('COHERE_API_KEY')
    if not key:
        raise RuntimeError('COHERE_API_KEY not set')
    url = 'https://api.cohere.ai/embed'
    headers = { 'Authorization': f'Bearer {key}', 'Content-Type':'application/json' }
    model = os.getenv('COHERE_EMBEDDING_MODEL', 'small')
    payload = { 'model': model, 'texts': [text] }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    emb = data.get('embeddings', [])[0]
    return np.array(emb, dtype='float32')


def _call_openai(text: str):
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY not set')
    url = 'https://api.openai.com/v1/embeddings'
    headers = { 'Authorization': f'Bearer {key}', 'Content-Type':'application/json' }
    model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    payload = { 'model': model, 'input': text }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    emb = data['data'][0]['embedding']
    return np.array(emb, dtype='float32')


def _call_voyage(text: str):
    key = os.getenv('VOYAGE_API_KEY')
    if not key:
        raise RuntimeError('VOYAGE_API_KEY not set')
    url = 'https://api.voyage.ai/v1/embeddings'
    headers = { 'Authorization': f'Bearer {key}', 'Content-Type':'application/json' }
    payload = { 'input': text }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    emb = data['embedding'] if 'embedding' in data else data.get('data', [])[0].get('embedding')
    return np.array(emb, dtype='float32')


CALL_MAP = {
    'huggingface': _call_huggingface,
    'cohere': _call_cohere,
    'openai': _call_openai,
    'voyage': _call_voyage,
}


def get_embedding(text: str):
    last_err = None
    for p in PROVIDER_ORDER:
        fn = CALL_MAP.get(p.lower())
        if not fn:
            continue
        try:
            emb = fn(text)
            return emb
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f'No embeddings provider succeeded. Last error: {last_err}')
