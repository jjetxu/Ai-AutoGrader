# deepseek_grader.py
import json
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, TIMEOUT, MAX_WORKERS, SAVE_EVERY,
    GRADER_SYSTEM_PROMPT, GRADER_USER_PROMPT_TEMPLATE
)
from src.io_utils import save_json
from src.dify_client import get_session

def deepseek_chat(messages) -> str:
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": messages, "temperature": 0.2, "max_tokens": 1000}

    s = get_session()
    r = s.post(DEEPSEEK_BASE_URL, json=payload, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def parse_grade_response(raw: str) -> dict:
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return {"score": 0, "rationale": "", "error": f"Non-JSON output: {raw[:200]}"}
        grade = json.loads(m.group(0))
        score = max(0, min(5, int(grade.get("score", 0))))
        return {"score": score, "rationale": str(grade.get("rationale", "")).strip(), "error": None}
    except Exception as e:
        return {"score": 0, "rationale": "", "error": str(e)}

def grade_record(record: dict) -> dict:
    q = record["question"]
    expected = record.get("expected_answer", "")
    agent = record["result"]["answer"]

    messages = [
        {"role": "system", "content": GRADER_SYSTEM_PROMPT},
        {"role": "user", "content": GRADER_USER_PROMPT_TEMPLATE.format(
            question=q, expected_answer=expected, agent_answer=agent
        )}
    ]

    try:
        raw = deepseek_chat(messages)
        g = parse_grade_response(raw)
        return {
            "id": record.get("id"),
            "question": q,
            "expected_answer": expected,
            "agent_answer": agent,
            "score": g["score"],
            "rationale": g["rationale"],
            "error": g["error"],
        }
    except Exception as e:
        return {
            "id": record.get("id"),
            "question": q,
            "expected_answer": expected,
            "agent_answer": agent,
            "score": 0,
            "rationale": "",
            "error": str(e),
        }

def run_parallel_grading(successful_results: list, out_path: str, max_workers: int = MAX_WORKERS) -> list:
    grades = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(grade_record, r) for r in successful_results]
        try:
            for n, fut in enumerate(as_completed(futures), start=1):
                g = fut.result()
                grades.append(g)
                print(f"✓ [{n}/{len(successful_results)}] Score: {g['score']}/5 - {g['question']}")
                if n % SAVE_EVERY == 0:
                    save_json(out_path, grades)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt — saving partial grading results and stopping...")
        finally:
            save_json(out_path, grades)
    return grades
