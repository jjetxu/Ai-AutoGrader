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
MODEL_NAME = os.getenv("MODEL_NAME")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT"))
RETRIES = int(os.getenv("MAX_RETRIES"))

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
    pass


def make_session(retries=RETRIES, verify=True) -> requests.Session:
    pass


def get_session() -> requests.Session:
    pass


def dify_chat(session: requests.Session, query: str, user: str, inputs=None) -> dict:
    pass


def worker(i: int, qobj: dict, user_prefix: str) -> dict:
    pass


def run_parallel_suite(questions, user_prefix: str, out_path: str, max_workers: int = MAX_WORKERS):
    pass


if __name__ == "__main__":
    pass
    # records = load_json("master_multitagged.json")
    # questions_objective, questions_subjective = split_objective_subjective(records)

    # print("=== Parallel Dify API Test (objective: hard_rules) ===")
    # run_parallel_suite(
    #     questions_objective,
    #     user_prefix="objective",
    #     out_path=OUT_OBJECTIVE,
    #     max_workers=MAX_WORKERS,
    # )

    # print("\n=== Parallel Dify API Test (subjective: non-hard_rules) ===")
    # run_parallel_suite(
    #     questions_subjective,
    #     user_prefix="subjective",
    #     out_path=OUT_SUBJECTIVE,
    #     max_workers=MAX_WORKERS,
    # )