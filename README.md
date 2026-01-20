# AI-AutoGrader

An automated evaluation framework for testing and grading AI agent responses. Designed to work with any LLM-based agent framework, AI-AutoGrader uses semantic analysis to evaluate answer quality against reference answers.

## Overview

AI-AutoGrader streamlines the process of:
1. **Querying** AI agents with structured question sets
2. **Filtering** successful responses
3. **Grading** responses using semantic similarity analysis
4. **Analyzing** overall performance metrics

Perfect for evaluating chatbots, RAG systems, agentic workflows, and any LLM-based application.

## Features

- **Parallel Processing** - Efficiently query and grade multiple responses simultaneously
- **Multi-Stage Pipeline** - Separate querying, filtering, and grading stages
- **Semantic Evaluation** - DeepSeek-powered semantic analysis for intelligent grading (1-5 scale)
- **Persistent Results** - Automatic checkpointing of intermediate and final results
- **Summary Statistics** - Average scores and detailed performance metrics
- **Framework Agnostic** - Works with Dify, LangChain, or any agent framework with API access

## Requirements

- Python 3.8+
- API access to your agent framework (e.g., Dify, custom LLM API)
- DeepSeek API key (for semantic grading)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-AutoGrader.git
cd AI-AutoGrader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file in the root directory:
```env
# Agent Framework API
API_KEY=your_api_key_here
API_BASE_URL=https://your-api-endpoint.com

# DeepSeek API (for grading)
DEEPSEEK_API_KEY=your_deepseek_key_here

# Optional Settings
MAX_WORKERS=4                    # Number of parallel workers
REQUEST_TIMEOUT=60               # API timeout in seconds
MAX_RETRIES=2                    # Retry attempts on failure
REQUEST_DELAY=0.5                # Delay between requests (seconds)
SAVE_EVERY=11                    # Save progress every N requests

# Output Paths
OUT_COMBINED=saved_results/dify_results_combined.json
OUT_SUCCESSFUL=saved_results/dify_results_successful.json
OUT_GRADES=saved_results/dify_grading_results.json
```

## Usage

### Basic Run

```bash
python src/main.py
```

This will:
1. Query your agent framework with questions from `input_questions/master_multitagged.json`
2. Filter for successful responses
3. Grade responses using semantic similarity
4. Display average scores

### Input Format

Your question file should be a JSON array:
```json
[
  {
    "question": "What is machine learning?",
    "expected_answer": "Machine learning is a subset of AI..."
  },
  ...
]
```

### Output Format

Grading results include:
```json
[
  {
    "question": "What is machine learning?",
    "expected_answer": "Machine learning is...",
    "agent_answer": "ML is a field of AI...",
    "score": 4,
    "rationale": "Answer captures main concepts but lacks detail"
  },
  ...
]
```

## Configuration

All settings are configurable via `.env` file. Key options:

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_WORKERS` | Parallel worker threads | 1 |
| `REQUEST_TIMEOUT` | API timeout (seconds) | 60 |
| `REQUEST_DELAY` | Delay between requests (seconds) | 0.5 |
| `SAVE_EVERY` | Save progress every N requests | 11 |

## Workflow

```
Input Questions
    ↓
[1] Parallel Query Agent Framework
    ↓
[2] Filter Successful Responses
    ↓
[3] Parallel Semantic Grading (DeepSeek)
    ↓
Output: Score & Rationale
```

## Extending the Framework

To adapt AI-AutoGrader for different frameworks:

1. **Modify API client** - Update `src/dify_client.py` to match your API
2. **Adjust data format** - Customize parsing in `src/io_utils.py`
3. **Custom grading** - Modify `GRADER_SYSTEM_PROMPT` in `src/config.py`

## License

MIT
