import os
import json
import time
import requests
import threading
import re
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

# CONFIG
# ---- Dify API ----
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 60))   # Request timeout in seconds (default: 60)
RETRIES = int(os.getenv("MAX_RETRIES", 2))          # Maximum number of retry attempts (default: 2)

# ---- DeepSeek API (for grading) ----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"

# ---- Threading & Output Configuration ----
MAX_WORKERS = 4                                      # Maximum number of parallel worker threads
SAVE_EVERY = 11                                       # Save results after every N completed requests
OUT_COMBINED = "saved_results/dify_results_combined.json"   # Output file for combined test results

# ---- Grading Configuration ----
GRADER_SYSTEM_PROMPT = "You are an expert evaluator tasked with assessing the quality of AI-generated answers based on their relevance, accuracy, and completeness compared to a reference answer."

GRADER_USER_PROMPT_TEMPLATE = """
Please evaluate the following AI-generated answer against the reference answer for the given question.

**Question:**
{question}

**Reference Answer:**
{expected_answer}

**AI-Generated Answer:**
{agent_answer}

**Evaluation Criteria:**
- Assign a score from 0 to 10 (0 = completely wrong, 10 = perfect match)
- Focus on factual accuracy, relevance, and completeness
- Consider whether the AI answer addresses all key points from the reference
- Be lenient on wording differences if the meaning is the same

**Your response must be in the following JSON format:**
{{"score": YOUR_SCORE, "rationale": "YOUR_EXPLANATION"}}
"""


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
        return {"id": qobj.get("id"), "question": q, "expected_answer": qobj.get("reference_answer", ""), "result": dify_chat(q, user=user)}
    except requests.exceptions.RequestException as e:
        msg = str(e)
        if "NameResolutionError" in msg or "getaddrinfo failed" in msg:
            time.sleep(2)
            try:
                return {"id": qobj.get("id"), "question": q, "expected_answer": qobj.get("reference_answer", ""), "result": dify_chat(q, user=user)}
            except requests.exceptions.RequestException as e2:
                return {"id": qobj.get("id"), "question": q, "expected_answer": qobj.get("reference_answer", ""), "result": {"success": False, "error": str(e2)}}
        return {"id": qobj.get("id"), "question": q, "expected_answer": qobj.get("reference_answer", ""), "result": {"success": False, "error": msg}}


def filter_results(input_path, output_path):
    """
    Filters results to include only successful entries.
    
    Args:
        input_path (str): Path to the input results JSON file
        output_path (str): Path to save the filtered results
    """
    results = load_json(input_path)
    successful_results = [result for result in results if result["result"].get("success", False)]
    save_json(output_path, successful_results)
    print(f"Filtered {len(results)} results to {len(successful_results)} successful ones")
    print(f"Saved to {output_path}")
    return successful_results


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


# ========== GRADING FUNCTIONS ==========

def api_post(base_url, endpoint, headers, payload) -> dict:
    """
    Generic API POST helper to reduce duplication between Dify/DeepSeek calls.
    
    Args:
        base_url (str): Base URL of the API
        endpoint (str): API endpoint path
        headers (dict): Request headers
        payload (dict): Request payload
        
    Returns:
        dict: API response data
        
    Raises:
        requests.exceptions.RequestException: If the API request fails
    """
    s = get_session()
    url = f"{base_url}{endpoint}"
    r = s.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def deepseek_chat(messages) -> str:
    """
    Sends a chat request to DeepSeek API and returns the assistant's response.
    
    Args:
        messages (list): List of chat messages in OpenAI format
        
    Returns:
        str: The assistant's response content
        
    Raises:
        requests.exceptions.RequestException: If the API request fails
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1000
    }
    response = api_post(DEEPSEEK_BASE_URL, "", headers, payload)
    return response["choices"][0]["message"]["content"].strip()


def parse_grade_response(raw_response):
    """
    Parses and validates JSON response from DeepSeek grader.
    
    Args:
        raw_response (str): Raw API response containing JSON
        
    Returns:
        dict: Parsed grade with score, rationale, and optional error
    """
    try:
        # Extract JSON from response using regex
        match = re.search(r"\{.*\}", raw_response, re.S)
        if not match:
            return {"score": 0, "rationale": "", "error": f"Non-JSON output: {raw_response[:200]}"}
        
        # Parse JSON
        grade = json.loads(match.group(0))
        
        # Validate score
        score = int(grade.get("score", 0))
        score = max(0, min(10, score))  # Ensure score is between 0-10
        
        return {
            "score": score,
            "rationale": grade.get("rationale", "").strip(),
            "error": None
        }
    except Exception as e:
        return {"score": 0, "rationale": "", "error": str(e)}


def compare_expected_response(record):
    """
    Compares AI-generated answer with expected answer using DeepSeek grading.
    
    Args:
        record (dict): Test record containing question, expected answer, and AI response
        
    Returns:
        dict: Grading result with question, score, rationale, and original data
    """
    question = record["question"]
    expected_answer = record["expected_answer"]
    agent_answer = record["result"]["answer"]
    
    # Prepare messages for DeepSeek grader
    messages = [
        {"role": "system", "content": GRADER_SYSTEM_PROMPT},
        {"role": "user", "content": GRADER_USER_PROMPT_TEMPLATE.format(
            question=question,
            expected_answer=expected_answer,
            agent_answer=agent_answer
        )}
    ]
    
    try:
        # Get grading response
        raw_grade = deepseek_chat(messages)
        
        # Parse and validate grading
        grade = parse_grade_response(raw_grade)
        
        return {
            "id": record.get("id"),
            "question": question,
            "expected_answer": expected_answer,
            "agent_answer": agent_answer,
            "score": grade["score"],
            "rationale": grade["rationale"],
            "error": grade["error"]
        }
    except Exception as e:
        return {
            "id": record.get("id"),
            "question": question,
            "expected_answer": expected_answer,
            "agent_answer": agent_answer,
            "score": 0,
            "rationale": "",
            "error": str(e)
        }


def grading_worker(record):
    """
    Worker function for processing individual grading tasks.
    
    Args:
        record (dict): Test record to grade
        
    Returns:
        dict: Grading result
    """
    return compare_expected_response(record)


def run_parallel_grading(successful_results, out_path, max_workers=MAX_WORKERS):
    """
    Runs parallel grading on successful results using ThreadPoolExecutor.
    
    Args:
        successful_results (list): List of successful test results to grade
        out_path (str): Path to save grading results
        max_workers (int): Maximum number of parallel worker threads
        
    Returns:
        list: Grading results
    """
    grading_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(grading_worker, record) for record in successful_results]
        
        try:
            for n, fut in enumerate(as_completed(futures), start=1):
                grade_rec = fut.result()
                grading_results.append(grade_rec)
                
                score = grade_rec["score"]
                q = grade_rec["question"]
                print(f"✓ [{n}/{len(successful_results)}] Score: {score}/10 - {q}")
                
                if n % SAVE_EVERY == 0:
                    save_json(out_path, grading_results)
        
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt — saving partial grading results and stopping...")
        finally:
            save_json(out_path, grading_results)
            print(f"Saved {len(grading_results)} grading results to {out_path}")
    
    return grading_results


if __name__ == "__main__":
    # ========== QUERYING SECTION ==========
    # You can comment out this entire section to skip querying and load existing results
    successful_results = None
    
    try:
        # ensure API can connect before running tests
        if not test_api_connection():
            print("Cannot connect to API. Exiting...")
            exit(1)
            
        # load test questions from JSON file
        records = load_json("input_questions/master_multitagged.json")

        # run tests for agent response
        print("=== Parallel Dify Querying ===")
        results = run_parallel_suite(records, "auto", OUT_COMBINED, MAX_WORKERS)

        # filter successful results and save to a separate file
        print("\n=== Filtering Results ===")
        successful_results = [result for result in results if result["result"].get("success", False)]
        filtered_out_path = "saved_results/dify_results_successful.json"
        save_json(filtered_out_path, successful_results)
        print(f"Saved {len(successful_results)} successful results to {filtered_out_path}")
        print(f"Original file {OUT_COMBINED} contains all {len(results)} results")
    except Exception as e:
        print(f"Querying failed with error: {e}")
        print("Falling back to loading existing successful results...")
        successful_results = None
    
    # ========== GRADING SECTION ==========
    # Load successful results if not obtained from querying
    if successful_results is None:
        print("\n=== Loading Existing Results ===")
        try:
            successful_results = load_json("saved_results/dify_results_successful.json")
            print(f"✓ Loaded {len(successful_results)} successful results from existing file")
        except FileNotFoundError:
            print("ERROR: Could not find 'saved_results/dify_results_successful.json'")
            print("Please either:")
            print("1. Uncomment the querying section to generate results")
            print("2. Ensure the successful results file exists")
            exit(1)
    
    # compare results with expected answers using DeepSeek grading
    print("\n=== Parallel Grading ===")
    grading_results = run_parallel_grading(
        successful_results,
        "saved_results/dify_grading_results.json",
        MAX_WORKERS
    )

    # Calculate and display overall statistics
    if grading_results:
        total_score = sum(r["score"] for r in grading_results)
        average_score = total_score / len(grading_results)
        print(f"\n=== Grading Summary ===")
        print(f"Total Questions Graded: {len(grading_results)}")
        print(f"Average Score: {average_score:.2f}/10")
        print(f"Total Score: {total_score}/{len(grading_results)*10}")
        print(f"Grading results saved to: saved_results/dify_grading_results.json")
