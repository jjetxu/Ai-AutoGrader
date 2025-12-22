# runner.py
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import MAX_WORKERS, SAVE_EVERY
from src.io_utils import save_json
from src.dify_client import dify_chat

def worker(i: int, qobj: dict, user_prefix: str) -> dict:
    q = qobj["question"]
    user = f"{user_prefix}-{i:03d}"
    expected = qobj.get("reference_answer", "")

    def fail(err: str) -> dict:
        return {"id": qobj.get("id"), "question": q, "expected_answer": expected,
                "result": {"success": False, "error": err}}

    try:
        res = dify_chat(q, user=user)
        return {"id": qobj.get("id"), "question": q, "expected_answer": expected, "result": res}
    except requests.exceptions.RequestException as e:
        msg = str(e)
        if "NameResolutionError" in msg or "getaddrinfo failed" in msg:
            time.sleep(2)
            try:
                res = dify_chat(q, user=user)
                return {"id": qobj.get("id"), "question": q, "expected_answer": expected, "result": res}
            except requests.exceptions.RequestException as e2:
                return fail(str(e2))
        return fail(msg)

def run_parallel_suite(questions: list, user_prefix: str, out_path: str, max_workers: int = MAX_WORKERS) -> list:
    results = []
    futures = []
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

    return results
