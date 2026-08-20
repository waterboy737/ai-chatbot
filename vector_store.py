import os
import sqlite3
import json
import numpy as np
from typing import List, Dict

# Simple file-backed vector store using numpy for vectors and sqlite for metadata

class VectorStore:
    def __init__(self, index_path: str = './data/faiss.index'):
        self.index_path = index_path
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self.db_path = index_path + '.meta.db'
        self._load()

    def _load(self):
        # metadata DB
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, meta TEXT, path TEXT)''')
        self.conn.commit()
        # vectors stored as .npy files per id in same dir
        self.vec_dir = os.path.join(os.path.dirname(self.index_path), 'vectors')
        os.makedirs(self.vec_dir, exist_ok=True)

    def add(self, id: str, vector: np.ndarray, meta: dict, text: str):
        vec_file = os.path.join(self.vec_dir, f'{id}.npy')
        np.save(vec_file, vector.astype('float32'))
        c = self.conn.cursor()
        c.execute('REPLACE INTO docs (id, meta, path) VALUES (?,?,?)', (id, json.dumps({'meta':meta,'text':text}), vec_file))
        self.conn.commit()

    def _iter_all(self):
        c = self.conn.cursor()
        for row in c.execute('SELECT id, meta, path FROM docs'):
            id, meta_json, path = row
            obj = json.loads(meta_json)
            vec = np.load(path)
            yield { 'id': id, 'meta': obj.get('meta',{}), 'text': obj.get('text',''), 'vector': vec }

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        # brute-force cosine similarity
        items = list(self._iter_all())
        if len(items) == 0:
            return []
        mats = np.stack([it['vector'] for it in items])
        # normalize
        q = query_vector.astype('float32')
        def norm(x):
            n = np.linalg.norm(x)
            return x / (n+1e-10)
        qn = norm(q)
        matsn = np.array([norm(r) for r in mats])
        sims = matsn.dot(qn)
        idx = sims.argsort()[::-1][:top_k]
        results = []
        for i in idx:
            it = items[i]
            results.append({ 'id': it['id'], 'meta': it['meta'], 'text': it['text'], 'score': float(sims[i]) })
        return results

