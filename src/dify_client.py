# dify_client.py
import time
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import API_KEY, API_BASE_URL, TIMEOUT, RETRIES

_thread_local = threading.local()

def _make_session(retries: int = RETRIES) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = _make_session()
    return _thread_local.session

def test_api_connection() -> bool:
    try:
        s = _make_session(retries=1)
        url = f"{API_BASE_URL}/chat-messages"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        s.head(url, headers=headers, timeout=5)
        print("✓ API endpoint is reachable")
        return True
    except Exception as e:
        print(f"✗ API connectivity test failed: {e}")
        print(f"  URL: {API_BASE_URL}")
        return False

def dify_chat(query: str, user: str, inputs=None) -> dict:
    s = get_session()
    url = f"{API_BASE_URL}/chat-messages"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Connection": "close",   # IMPORTANT for unstable tunnels
    }
    payload = {"inputs": inputs or {}, "query": query, "response_mode": "blocking", "user": user}

    t0 = time.time()
    r = s.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    dt = time.time() - t0

    if r.status_code >= 400:
        # <-- This tells you the real reason
        raise requests.HTTPError(f"{r.status_code} {r.reason} | body={r.text[:1000]}", response=r)

    data = r.json()
    return {"success": True, "time": dt, "answer": data.get("answer"), "conversation_id": data.get("conversation_id")}

