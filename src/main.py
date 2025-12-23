# main.py
from src.dify_client import test_api_connection
from src.io_utils import load_json, save_json, filter
from src.runner import run_parallel_suite
from src.grader import run_parallel_grading
from src.config import OUT_COMBINED, OUT_SUCCESSFUL, OUT_GRADES, MAX_WORKERS

def main():
    successful_results = None

    # Querying section
    try:
        if not test_api_connection():
            print("Cannot connect to API. Exiting...")
            return

        records = load_json("input_questions/master_multitagged.json")
        print("=== Parallel Dify Querying ===")
        results = run_parallel_suite(records, "auto", OUT_COMBINED, MAX_WORKERS)
        # results = load_json(OUT_COMBINED)
        print("\n=== Filtering Results ===")
        successful_results = filter(results, "success", "==", True)
        save_json(OUT_SUCCESSFUL, successful_results)
        print(f"Saved {len(successful_results)} successful results to {OUT_SUCCESSFUL}")
        print(f"Original file {OUT_COMBINED} contains all {len(results)} results")

    except Exception as e:
        print(f"Querying failed with error: {e}")
        successful_results = None

    # Grading section
    if successful_results is None:
        print("\n=== Loading Existing Results ===")
        successful_results = load_json(OUT_SUCCESSFUL)
        print(f"✓ Loaded {len(successful_results)} successful results from existing file")

    print("\n=== Parallel Grading ===")
    grading_results = run_parallel_grading(successful_results, OUT_GRADES, MAX_WORKERS)

    if grading_results:
        avg = sum(r["score"] for r in grading_results) / len(grading_results)
        print("\n=== Grading Summary ===")
        print(f"Total Questions Graded: {len(grading_results)}")
        print(f"Average Score: {avg:.2f}/5")

if __name__ == "__main__":
    main()
