# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ---- Dify API ----
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 60))
RETRIES = int(os.getenv("MAX_RETRIES", 2))

# ---- DeepSeek API ----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"

# ---- Threading & Output ----
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 1))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", 11))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", 0.5))
OUT_COMBINED = os.getenv("OUT_COMBINED", "saved_results/dify_results_combined.json")
OUT_SUCCESSFUL = os.getenv("OUT_SUCCESSFUL", "saved_results/dify_results_successful.json")
OUT_GRADES = os.getenv("OUT_GRADES", "saved_results/dify_grading_results.json")

# ---- Grader prompts ----
GRADER_SYSTEM_PROMPT = "你作为一个语义分析大师，具有较强的语义理解能力，请帮我分析针对一个问题我的答案和参考答案之间的相似性，并给出评分"

GRADER_USER_PROMPT_TEMPLATE = """
针对一个问题，我提供了参考答案和我的回答，请评判我的回答是否合适、是否准确，给出一个评分1-5分，

**Question:**
{question}

**Reference Answer:**
{expected_answer}

**AI-Generated Answer:**
{agent_answer}

**Evaluation Criteria:**
5分表示回答和参考答案非常切合，和参考答案相同的意思；
5分表示回答和参考答案很契合，能够表达的基本相同的含义；
3分表示回答和参考答案相近，有较大的词语相似；
2分表示回答和参考答案差异较大，只有很少的部分或者词语相似；
1分表示回答和参考答案不具有相近性；

**Your response must be in the following JSON format:**
{{"score": YOUR_SCORE, "rationale": "YOUR_EXPLANATION"}}
"""
