import os
import json
import time
import requests
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# CONFIG
# ---- API ----
API_KEY = os.getenv("PERSONAL_API_KEY")
API_BASE_URL = os.getenv("PERSONAL_API_BASE_URL")
TIMEOUT = os.getenv("REQUEST_TIMEOUT")
RETRIES = os.getenv("MAX_RETRIES")

# ---- Threading ----
MAX_WORKERS = 3
SAVE_EVERY = 1           # save after every completed request
OUT_OBJECTIVE = "dify_results_objective.json"
OUT_SUBJECTIVE = "dify_results_subjective.json"


def save_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_objective_subjective(records):
    obj, sub = [], []
    for r in records:
        (obj if "hard_rules" in r.get("eval", []) else sub).append(r)
    return obj, sub


def make_session(retries=RETRIES, verify=True) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=RETRIES,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


_thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = make_session()
    return _thread_local.session


def dify_chat(query: str, user: str, inputs=None) -> dict:
    s = get_session()
    url = f"{API_BASE_URL}/chat-messages"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": inputs or {}, "query": query, "response_mode": "blocking", "user": user}

    t0 = time.time()
    r = s.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    dt = time.time() - t0
    r.raise_for_status()

    data = r.json()
    return {"success": True, "time": dt, "answer": data.get("answer"), "conversation_id": data.get("conversation_id")}


def worker(i: int, qobj: dict, user_prefix: str) -> dict:
    q = qobj["question"]
    user = f"{user_prefix}-{i:03d}"
    try:
        return {"id": qobj.get("id"), "question": q, "result": dify_chat(q, user=user)}
    except requests.exceptions.RequestException as e:
        msg = str(e)
        if "NameResolutionError" in msg or "getaddrinfo failed" in msg:
            time.sleep(2)
            try:
                return {"id": qobj.get("id"), "question": q, "result": dify_chat(q, user=user)}
            except requests.exceptions.RequestException as e2:
                return {"id": qobj.get("id"), "question": q, "result": {"success": False, "error": str(e2)}}
        return {"id": qobj.get("id"), "question": q, "result": {"success": False, "error": msg}}


def run_parallel_suite(questions, user_prefix: str, out_path: str, max_workers: int = MAX_WORKERS):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, i, q, user_prefix) for i, q in enumerate(questions)]
        try:
            for n, fut in enumerate(as_completed(futures), start=1):
                rec = fut.result()
                results.append(rec)

                res = rec["result"]
                ok = "✓" if res.get("success") else "✗"
                msg = res.get("answer") or res.get("error") or ""
                t = res.get("time")
                t_str = f"{t:.2f}s" if isinstance(t, (int, float)) else ""
                print(f"{ok} [{n}/{len(questions)}] {rec['question']} {t_str} {msg}")

                if n % SAVE_EVERY == 0:
                    save_json(out_path, results)

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt — saving partial results and stopping...")
        finally:
            save_json(out_path, results)
            print(f"Saved {len(results)} results to {out_path}")

    return results


if __name__ == "__main__":
    records = load_json("master_multitagged.json")
    questions_objective, questions_subjective = split_objective_subjective(records)

    print("=== Parallel Dify API Test (objective) ===")
    run_parallel_suite(questions_objective, "objective", OUT_OBJECTIVE, MAX_WORKERS)

    print("\n=== Parallel Dify API Test (subjective) ===")
    run_parallel_suite(questions_subjective, "subjective", OUT_SUBJECTIVE, MAX_WORKERS)