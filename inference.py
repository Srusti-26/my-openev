"""
Inference Script for EmailTriageEnv - OpenEnv Compatible

Mandatory requirements:
- Defines API_BASE_URL, MODEL_NAME, HF_TOKEN environment variables
- Uses OpenAI Client for all LLM calls
- Emits structured stdout logs with [START], [STEP], [END] format
- Each task returns score in [0, 1] range
- Must complete without error

Environment variables:
  API_BASE_URL   - LLM API endpoint (default: HuggingFace Inference API)
  MODEL_NAME     - Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       - HuggingFace token for inference
"""

import os
import sys
import json
import textwrap
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

from data.loader import load_emails
from env.email_env import EmailTriageEnv, ACTION_NAMES
from tasks.easy_task import EasyTask
from tasks.medium_task import MediumTask
from tasks.hard_task import HardTask


# ============================================================================
# Configuration from environment
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN or HF_API_TOKEN not set. Please set the environment variable.", file=sys.stderr)
    sys.exit(1)

# ============================================================================
# Task mapping
# ============================================================================

TASK_MAP = {
    "easy": EasyTask,
    "medium": MediumTask,
    "hard": HardTask,
}

# ============================================================================
# System prompt for the agent
# ============================================================================

SYSTEM_PROMPT = textwrap.dedent("""
    You are an intelligent email triage assistant. Your job is to analyze emails
    and decide the best action to take.
    
    Available actions:
      0 - classify_spam    (mark as spam, do not reply)
      1 - classify_urgent  (mark as urgent, needs immediate attention)
      2 - classify_normal  (mark as normal priority)
      3 - reply            (compose a professional reply)
      4 - escalate         (forward to a senior team member)
    
    For each email, respond with a JSON object like:
    {
      "action_id": <0-4>,
      "reply_text": "<reply if action 3 or 4, else empty>"
    }
    
    Reply with ONLY valid JSON, no markdown or extra text.
""").strip()


# ============================================================================
# LLM client and inference functions
# ============================================================================

def init_client() -> OpenAI:
    """Initialize OpenAI client with HuggingFace endpoint."""
    return OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )


def call_llm(client: OpenAI, observation: dict) -> str:
    """Call LLM to get action for current observation."""
    user_message = textwrap.dedent(f"""
        Email from: {observation['sender']}
        Subject: {observation['subject']}
        Body: {observation['body'][:300]}
        
        Respond with JSON only.
    """).strip()
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", file=sys.stderr)
        return '{"action_id": 2, "reply_text": ""}'


def parse_action(raw_response: str) -> dict:
    """Parse LLM response into action dict."""
    try:
        text = raw_response.strip()
        
        # Remove markdown code fences if present
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
        
        text = text.strip()
        action = json.loads(text)
        
        # Ensure action_id is in valid range
        action_id = int(action.get("action_id", 2))
        if action_id < 0 or action_id > 4:
            action_id = 2
        
        return {
            "action_id": action_id,
            "reply_text": str(action.get("reply_text", "")),
        }
    except json.JSONDecodeError:
        return {"action_id": 2, "reply_text": ""}


# ============================================================================
# Main evaluation loop
# ============================================================================

def run_task(
    client: OpenAI,
    task_name: str,
    dataset: list[dict],
    episodes: int = 2,
    seed: int = 42,
) -> float:
    """
    Run one task for multiple episodes and return average score.
    
    Returns:
        score in [0.0, 1.0]
    """
    if task_name not in TASK_MAP:
        print(f"[ERROR] Unknown task: {task_name}", file=sys.stderr)
        return 0.0
    
    task_class = TASK_MAP[task_name]
    total_reward = 0.0
    num_complete = 0
    
    for ep in range(episodes):
        task = task_class()
        env = EmailTriageEnv(
            task=task,
            dataset=dataset,
            seed=seed + ep,
            max_steps=5,
            render_mode="json",
        )
        
        obs, info = env.reset()
        step_num = 1
        episode_reward = 0.0
        
        # Log episode start
        print(f"[START] task={task_name} env=email_triage_env model={MODEL_NAME}")
        
        done = False
        
        while not done and step_num <= 5:
            try:
                # Get action from LLM
                raw_action = call_llm(client, obs)
                action = parse_action(raw_action)
                
                # Step environment
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Log step with reward
                action_name = ACTION_NAMES.get(action["action_id"], "unknown")
                print(
                    f"[STEP] step={step_num} action={action_name} "
                    f"reward={reward:.2f} done={str(done).lower()} error=null"
                )
                
                step_num += 1
                
            except Exception as e:
                error_msg = str(e).replace('"', "'")
                print(
                    f"[STEP] step={step_num} action=error "
                    f"reward=0.00 done=true error=\"{error_msg}\""
                )
                done = True
        
        total_reward += episode_reward
        num_complete += 1
        
        # Log episode end
        success = (episode_reward > 0.1)
        print(
            f"[END] success={str(success).lower()} steps={step_num} "
            f"score={episode_reward:.2f} rewards={episode_reward:.2f}"
        )
    
    # Return average score normalized to [0, 1]
    avg_reward = total_reward / num_complete if num_complete > 0 else 0.0
    return min(1.0, max(0.0, avg_reward))


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """Run inference on all tasks."""
    print(f"[INFO] Using model: {MODEL_NAME}", file=sys.stderr)
    print(f"[INFO] API base URL: {API_BASE_URL}", file=sys.stderr)
    
    # Load dataset
    try:
        dataset = load_emails()
        print(f"[INFO] Loaded {len(dataset)} emails", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize LLM client
    try:
        client = init_client()
        print(f"[INFO] Initialized OpenAI client", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Failed to initialize client: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run all tasks
    scores = {}
    for task_name in ["easy", "medium", "hard"]:
        print(f"[INFO] Running task: {task_name}", file=sys.stderr)
        try:
            score = run_task(client, task_name, dataset, episodes=1, seed=42)
            scores[task_name] = score
            print(f"[INFO] Task {task_name} score: {score:.4f}", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Task {task_name} failed: {e}", file=sys.stderr)
            scores[task_name] = 0.0
    
    # Print summary
    avg_score = sum(scores.values()) / len(scores)
    print(f"\n[SUMMARY]", file=sys.stderr)
    for task_name, score in scores.items():
        print(f"  {task_name}: {score:.4f}", file=sys.stderr)
    print(f"  average: {avg_score:.4f}", file=sys.stderr)
    
    return 0 if avg_score > 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
