from src.io_utils import load_json, save_json, filter


data = load_json("saved_results/dify_grading_results.json")
filtered_data = filter(data, "score", "<=", 3)
print(filtered_data)
save_json("saved_results/dify_improvements.json", filtered_data)
print(f"Filtered {len(filtered_data)} results with score <= 3")
