---
sdk: docker
---

# EmailTriageEnv — OpenEnv-Compatible Environment

A Gymnasium-compatible and OpenEnv-compliant RL environment for LLM email-triage agents.
This version uses the **HuggingFace Inference API** for model execution.

##  Overview

**EmailTriageEnv** simulates real-world email triage tasks with three difficulty levels:

- **Easy**: Binary email classification (spam / urgent / normal)
- **Medium**: Classification + professional reply generation
- **Hard**: Multi-step reasoning with thread context and escalation logic

All graders are **deterministic** — the same actions always yield the same scores.

## Features

✅ **OpenEnv-Compliant** — Full HTTP API with typed Pydantic models
✅ **Gymnasium-Compatible** — Standard RL environment interface
✅ **Real-World Tasks** — Email triage, classification, and response generation
✅ **Three Difficulty Levels** — Easy → Medium → Hard curriculum
✅ **Deterministic Grading** — Reproducible scores via rule-based graders
✅ **Containerized** — Works seamlessly on Hugging Face Spaces

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a HuggingFace API token

1. Go to <https://huggingface.co/settings/tokens>
2. Create a token with **read** scope (free account works)
3. Export it:

```bash
export HF_API_TOKEN=hf_xxxxxxxxxxxx
```

### 3a. Run the inference script

```bash
# Easy task — classify spam / urgent / normal (3 episodes)
python inference.py --task easy --episodes 3

# Medium task — classify + reply
python inference.py --task medium --episodes 3 --render

# Hard task — full multi-step reasoning
python inference.py --task hard --episodes 2 --render --seed 123

# Use a different HF model
python inference.py --task easy --model HuggingFaceH4/zephyr-7b-beta
```

### 3b. Run the HTTP API server (OpenEnv)

```bash
# Start the FastAPI server on port 8000
python -m uvicorn server:app --host 0.0.0.0 --port 8000

# Then use curl or any HTTP client:
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'

curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_id": 2, "reply_text": ""}}'
```

### 3c. Run with Docker

```bash
# Build the image
docker build -t email_triage_env .

# Run the API server
docker run -e HF_API_TOKEN=$HF_API_TOKEN -p 8000:8000 email_triage_env

# Run the inference script (3 episodes of easy task)
docker run -e HF_API_TOKEN=$HF_API_TOKEN --entrypoint python email_triage_env \
  inference.py --task easy --episodes 3
```

---

## OpenEnv API Endpoints

The environment implements the OpenEnv specification with the following REST API:

| Endpoint | Method | Purpose |
|---|---|---|
| `/reset` | POST | Initialize a new episode |
| `/step` | POST | Submit action, get observation & reward |
| `/state` | GET | Get current environment state |
| `/seed` | POST | Set random seed |
| `/info` | GET | Get environment metadata |
| `/health` | GET | Health check |
| `/tasks` | GET | List available tasks |

### Example: Reset and Step

```bash
# Reset environment
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}' | jq .

# Submit action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_id": 2,
      "reply_text": ""
    }
  }' | jq .
```

### Data Models (Pydantic)

All API requests/responses use typed Pydantic models:

```python
from models import ObservationModel, ActionModel, StepResponseModel

# Observation
{
    "email_id": "email_001",
    "sender": "customer@example.com",
    "subject": "Account access issue",
    "body": "I cannot log into my account...",
    "thread_history": "[]",
    "metadata": "{}",
    "step_count": 0
}

# Action
{
    "action_id": 2,  # 0=spam, 1=urgent, 2=normal, 3=reply, 4=escalate
    "reply_text": ""
}

# Response
{
    "observation": {...},
    "reward": 1.0,
    "done": false,
    "info": {"grader_info": {...}}
}
```

---

## Task Definitions

### Easy: Email Classification (1 step)

The agent must classify each email as **spam**, **urgent**, or **normal**.

| Metric | Value |
|---|---|
| Max Steps | 1 |
| Reward | Binary (1.0 correct / 0.0 wrong) |
| Valid Actions | classify_spam, classify_urgent, classify_normal |

### Medium: Classification + Reply (2 steps)

Two-step episode: (1) classify the email, (2) generate an appropriate reply.

| Metric | Value |
|---|---|
| Max Steps | 2 |
| Reward | 40% classification accuracy + 60% reply quality |
| Valid Actions | All |
| Special | Spam emails terminate after step 1 (no reply) |

### Hard: Multi-Step Reasoning (3 steps)

Full reasoning with thread context:
1. Classify email (spam / urgent / normal)
2. Decide action (reply / escalate)
3. Compose appropriate reply or escalation memo

| Metric | Value |
|---|---|
| Max Steps | 3 |
| Reward | 20% classification + 20% action decision + 60% output quality |
| Reward Breakdown | Keyword match: 30%, Safety: 15%, Tone: 15% |
| Special | Uses thread history, VIP status, PII metadata |

---

## Graders (Reward Functions)

### ClassificationGrader

- **Type**: Accuracy-based
- **Scoring**: 1.0 if prediction == ground_truth, else 0.0
- **Deterministic**: ✓ Yes — no randomness
- **Used in**: Easy, Medium, Hard tasks

### ReplyGrader

- **Type**: Weighted keyword + relevance + length
- **Weights**: 70% keyword match, 20% relevance to body, 10% length (min 15 words)
- **Deterministic**: ✓ Yes — pure regex/string matching
- **Used in**: Medium, Hard tasks

### SafetyGrader

- **Type**: Rule-based PII + unsafe language detection
- **Scoring**: 
  - Deduct 0.5 per PII field leaked (SSN, credit card, phone, password)
  - Deduct 0.3 for unsafe phrases ("we guarantee", "legal action", etc.)
- **Deterministic**: ✓ Yes — hardcoded patterns
- **Used in**: Hard task

All graders are **deterministic** — reproducing the same actions yields the same scores.

---

## Recommended Free Models

| Model | Pros | Cons |
|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.3` | **Fast**, good JSON compliance | Slightly lower quality |
| `HuggingFaceH4/zephyr-7b-beta` | Excellent instruction following | Moderate speed |
| `meta-llama/Meta-Llama-3-8B-Instruct` | Highest quality | Requires HF license acceptance |

All are available on the HF free-tier Inference API (with occasional rate limiting).

---

## Run Tests (no API key required)

The test suite is fully deterministic — it uses rule-based agents and never calls the LLM.

```bash
pytest tests/ -v
# or with coverage:
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Project Structure

```
.
├── app.py                        ← Gradio UI (legacy)
├── inference.py                  ← LLM agent runner (HuggingFace Inference API)
├── server.py                     ← FastAPI OpenEnv HTTP API
├── models.py                     ← Pydantic data models
├── openenv.yaml                  ← OpenEnv specification
├── Dockerfile                    ← Container definition
├── requirements.txt              ← Python dependencies
│
├── env/
│   ├── __init__.py
│   └── email_env.py             ← Gymnasium environment
│
├── tasks/
│   ├── __init__.py
│   ├── base_task.py             ← Abstract task base class
│   ├── easy_task.py             ← Classify only
│   ├── medium_task.py           ← Classify + reply
│   └── hard_task.py             ← Full multi-step reasoning
│
├── graders/
│   ├── __init__.py
│   ├── classification_grader.py  ← Label matching
│   ├── reply_grader.py          ← Keyword + relevance + length
│   └── safety_grader.py         ← PII & unsafe language detection
│
├── data/
│   ├── __init__.py
│   ├── emails.json              ← Email dataset (15 examples)
│   ├── app/                     ← Application-specific data
│   └── loader.py               ← Dataset loading utilities
│
└── tests/
    ├── __init__.py
    └── test_env.py              ← Gymnasium & grader tests
```

---

## Deployment on Hugging Face Spaces

### Option 1: Clone and Deploy

```bash
# Clone the Spaces repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/my-openenv
cd my-openenv

# (optional) Create a .env file with your HF token:
echo "HF_API_TOKEN=hf_xxxx" > .env
```

Then edit the files in the Space's web editor on Hugging Face, or manage via git.

### Option 2: Use Docker Image

Specify in your Space's config:
```yaml
title: EmailTriageEnv API
description: OpenEnv-compatible email triage environment
app_file: null  # Use Docker instead
docker:
  image: email_triage_env:latest
```

---

## Baseline Performance

Sample runs with `mistralai/Mistral-7B-Instruct-v0.3` (5 episodes each, seed=42):

| Task | Episodes | Avg Reward | Notes |
|---|---|---|---|
| Easy | 5 | 0.82 | Good classification accuracy |
| Medium | 3 | 0.71 | Classification 0.8, reply quality 0.65 |
| Hard | 2 | 0.68 | Reasoning + safety checks impact score |

*(Baselines depend on model choice and random seed)*

To reproduce:
```bash
export HF_API_TOKEN=hf_xxxx
python inference.py --task easy --episodes 5 --seed 42
python inference.py --task medium --episodes 3 --seed 42
python inference.py --task hard --episodes 2 --seed 42
```

---

## Troubleshooting

**`ImportError: No module named 'models'`**
- Ensure you're running from the project root: `cd /path/to/my-openev`
- Or add to PYTHONPATH: `export PYTHONPATH=/path/to/my-openev:$PYTHONPATH`

**`HF_API_TOKEN not set`**
- Export it: `export HF_API_TOKEN=hf_xxxxxxxxxxxx`
- Or create a `.env` file and use `python-dotenv`: `from dotenv import load_dotenv; load_dotenv()`

**`401 Unauthorized`**
- Your token is wrong or expired
- Regenerate at <https://huggingface.co/settings/tokens>

**`503 Model loading`**
- The model is cold-starting on HF's free tier
- The script waits automatically and retries
- (Upgrade to HF Pro to avoid cold starts)

**`429 Rate limited`**
- You've hit the free-tier quota
- The script backs off and retries with exponential delay
- Upgrade to HF Pro for higher limits

**`JSON parse warning`**
- The model returned non-JSON text
- The script falls back to `classify_normal`
- Try a different model or lower `--episodes` to reduce context length

---

## Citation

If you use this environment in research, please cite:

```bibtex
@software{email_triage_env_2024,
  author = {AI Systems Engineer},
  title = {EmailTriageEnv: OpenEnv-Compatible Email Triage RL Environment},
  year = {2024},
  url = {https://huggingface.co/spaces/srusti-26/my-openenv}
}
```

---

## License

MIT License — See LICENSE file for details
