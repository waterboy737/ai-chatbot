import os
from dotenv import load_dotenv

# Load .env early so provider modules that read env at import time see values
load_dotenv()

from flask import Flask, render_template, request, jsonify
import providers.llm as llm_provider
import providers.embeddings as emb_provider
from vector_store import VectorStore

app = Flask(__name__, template_folder='templates')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev')

# Simple vector store instance (file-backed)
vs = VectorStore(index_path=os.getenv('FAISS_INDEX_PATH', './data/faiss.index'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    message = data.get('message', '')
    history = data.get('history', [])

    if not message or not message.strip():
        return jsonify({"error": "empty message"}), 400

    # Basic RAG flow: find top-k docs, send with prompt to LLM
    try:
        query_embedding = emb_provider.get_embedding(message)
    except Exception as e:
        return jsonify({"error": f"embedding error: {e}"}), 500

    docs = vs.search(query_embedding, top_k=4)

    context_text = "\n\n".join([f"Source: {d['meta'].get('source','unknown')}\n{d['text']}" for d in docs])

    prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's question.\n\n"
        f"CONTEXT:\n{context_text}\n\nUSER:\n{message}"
    )

    # Call LLM provider
    try:
        llm_reply = llm_provider.generate(prompt)
    except Exception as e:
        return jsonify({"error": f"LLM provider error: {e}"}), 500

    return jsonify({"reply": llm_reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
