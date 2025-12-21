import os
import json
import time
import requests
import threading
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

# CONFIG
# ---- API ----
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 60))   # Request timeout in seconds (default: 60)
RETRIES = int(os.getenv("MAX_RETRIES", 2))          # Maximum number of retry attempts (default: 2)

# ---- Threading & Output Configuration ----
MAX_WORKERS = 1                                      # Maximum number of parallel worker threads
SAVE_EVERY = 1                                       # Save results after every N completed requests
OUT_OBJECTIVE = "saved_results/dify_results_objective.json"   # Output file for objective test results
OUT_SUBJECTIVE = "saved_results/dify_results_subjective.json" # Output file for subjective test results


def save_json(path: str, data) -> None:
    """
    Saves data to a JSON file with atomic write operation.
    
    Uses temporary file + replace to ensure data integrity in case of interruptions.
    
    Args:
        path (str): Path to the output JSON file
        data: Data to be saved (must be JSON serializable)
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path: str):
    """
    Loads data from a JSON file.
    
    Args:
        path (str): Path to the input JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_objective_subjective(records):
    """
    Splits test records into objective and subjective categories.
    
    Objective questions are identified by having "hard_rules" in their evaluation tags.
    
    Args:
        records (list): List of test records to split
        
    Returns:
        tuple: (objective_questions, subjective_questions)
    """
    obj, sub = [], []
    for r in records:
        (obj if "hard_rules" in r.get("eval", []) else sub).append(r)
    return obj, sub


def test_api_connection():
    """Test if the API endpoint is reachable and responsive."""
    try:
        s = make_session(retries=1)
        # Send a simple HEAD request to test connectivity
        url = f"{API_BASE_URL}/chat-messages"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = s.head(url, headers=headers, timeout=5)
        print("✓ API endpoint is reachable")
        return True
    except Exception as e:
        print(f"✗ API connectivity test failed: {e}")
        print(f"  URL: {API_BASE_URL}")
        return False


def make_session(retries=RETRIES, verify=True) -> requests.Session:
    """
    Creates a requests Session with retry configuration.
    
    Configures session with retry logic for transient errors (429, 500-504).
    
    Args:
        retries (int): Maximum number of retry attempts
        verify (bool): Whether to verify SSL certificates
        
    Returns:
        requests.Session: Configured session object
    """
    s = requests.Session()
    retry = Retry(
        total=retries,
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
    """
    Gets a thread-local requests Session.
    
    Ensures each thread has its own session instance to avoid thread-safety issues.
    Creates a new session if one doesn't exist for the current thread.
    
    Returns:
        requests.Session: Thread-local configured session
    """
    if not hasattr(_thread_local, "session"):
        _thread_local.session = make_session()
    return _thread_local.session


def dify_chat(query: str, user: str, inputs=None) -> dict:
    """
    Sends a chat query to Dify API and returns the response.
    
    Args:
        query (str): The user's query to send to the AI model
        user (str): Unique identifier for the user (used for conversation tracking)
        inputs (dict, optional): Additional input parameters for the model
        
    Returns:
        dict: Response containing success status, answer, time taken, and conversation ID
        
    Raises:
        requests.exceptions.RequestException: If the API request fails
    """
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
    """
    Worker function for processing individual test questions.
    
    Handles sending queries to Dify API with retry logic for specific network errors.
    
    Args:
        i (int): Worker index (used for user ID generation)
        qobj (dict): Question object containing the test query
        user_prefix (str): Prefix for generating unique user IDs
        
    Returns:
        dict: Result containing question, ID, and API response
    """
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
    """
    Runs a parallel suite of test questions using ThreadPoolExecutor.
    
    Processes questions concurrently, handles keyboard interrupts gracefully,
    and saves results incrementally.
    
    Args:
        questions (list): List of question objects to test
        user_prefix (str): Prefix for generating unique user IDs
        out_path (str): Path to save the results JSON file
        max_workers (int): Maximum number of parallel worker threads
        
    Returns:
        list: All test results
    """
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
    if not test_api_connection():
        print("Cannot connect to API. Exiting...")
        exit(1)
        
    records = load_json("input_questions/master_multitagged.json")
    questions_objective, questions_subjective = split_objective_subjective(records)

    print("=== Parallel Dify API Test (objective) ===")
    run_parallel_suite(questions_objective, "objective", OUT_OBJECTIVE, MAX_WORKERS)

    print("\n=== Parallel Dify API Test (subjective) ===")
    run_parallel_suite(questions_subjective, "subjective", OUT_SUBJECTIVE, MAX_WORKERS)