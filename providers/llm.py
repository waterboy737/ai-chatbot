import os
import requests

LLM_PROVIDERS = [p.strip() for p in os.getenv('LLM_PROVIDERS', 'huggingface,openai').split(',') if p.strip()]


def _call_huggingface(prompt: str):
    hf_key = os.getenv('HUGGINGFACE_API_KEY')
    if not hf_key:
        raise RuntimeError('HUGGINGFACE_API_KEY not set')
    model = os.getenv('HF_LLM_MODEL', 'gpt2')
    url = f'https://api-inference.huggingface.co/models/{model}'
    headers = { 'Authorization': f'Bearer {hf_key}', 'Content-Type':'application/json' }
    payload = { 'inputs': prompt, 'options': { 'wait_for_model': True } }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # HF may return string or list with generated_text
    if isinstance(data, dict) and 'generated_text' in data:
        return data['generated_text']
    if isinstance(data, list) and len(data)>0 and 'generated_text' in data[0]:
        return data[0]['generated_text']
    # Some HF models return [{'generated_text':...}]
    if isinstance(data, dict) and 'error' in data:
        raise RuntimeError(data['error'])
    # Fallback: try to stringify
    return str(data)


def _call_openai(prompt: str):
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY not set')
    url = 'https://api.openai.com/v1/chat/completions'
    headers = { 'Authorization': f'Bearer {key}', 'Content-Type':'application/json' }
    model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    messages = [ { 'role':'system','content':'You are a helpful assistant.' }, { 'role':'user','content': prompt } ]
    payload = { 'model': model, 'messages': messages, 'temperature': 0.2 }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data['choices'][0]['message']['content']


CALL_MAP = {
    'huggingface': _call_huggingface,
    'openai': _call_openai,
}


def generate(prompt: str):
    last_err = None
    for p in LLM_PROVIDERS:
        fn = CALL_MAP.get(p.lower())
        if not fn:
            continue
        try:
            return fn(prompt)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f'No LLM provider succeeded. Last error: {last_err}')
