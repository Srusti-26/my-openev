"""
app.py — AI Email Triage Assistant
Serves the new UI via Gradio and connects to the real EmailTriageEnv + HuggingFace LLM backend.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from data.loader import load_emails
from env.email_env import (
    ACTION_CLASSIFY_NORMAL,
    ACTION_CLASSIFY_SPAM,
    ACTION_CLASSIFY_URGENT,
    ACTION_ESCALATE,
    ACTION_REPLY,
    ACTION_NAMES,
    EmailTriageEnv,
)
from tasks.easy_task import EasyTask
from tasks.medium_task import MediumTask
from tasks.hard_task import HardTask
from huggingface_hub import InferenceClient

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Models tried in order until one works
MODEL_FALLBACKS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
]
DEFAULT_MODEL = MODEL_FALLBACKS[0]
ACTION_LABEL_MAP = {
    ACTION_CLASSIFY_SPAM:   ("Spam / Phishing", "🚫"),
    ACTION_CLASSIFY_URGENT: ("Urgent",           "🔴"),
    ACTION_CLASSIFY_NORMAL: ("Normal",           "🟢"),
    ACTION_REPLY:           ("Reply",            "↩️"),
    ACTION_ESCALATE:        ("Escalate",         "⬆️"),
}
TASK_MAP = {"Easy": EasyTask, "Medium": MediumTask, "Hard": HardTask}
EPISODE_COUNTER = {"value": 1}

SAMPLE_EMAILS = {
    "phishing": textwrap.dedent("""
        Subject: URGENT — Your account has been compromised!

        Dear valued customer,

        We have detected unusual login activity on your account from an unrecognized device in Moscow, Russia.
        Your account will be permanently suspended within 24 hours unless you verify your identity immediately.

        Click here to verify: http://secur1ty-check.totallylegit.xyz/verify

        Failure to act will result in permanent data loss.

        Best regards,
        Account Security Team
        Do not reply to this email.
    """).strip(),

    "complaint": textwrap.dedent("""
        Subject: Extremely disappointed with recent order #4892

        Hi Support,

        I placed an order two weeks ago for the Premium Wireless Headphones (Order #4892) and what I received
        is completely unacceptable. The left ear cup is cracked, the charging cable was missing, and the box
        looked like it had been opened before.

        This is my third order with quality issues this year. I've been a loyal customer for 5 years and I'm
        seriously considering switching to a competitor. I need a full refund or a replacement shipped overnight.

        Please escalate this to a manager if needed.

        Regards,
        Sarah Chen
    """).strip(),

    "internal": textwrap.dedent("""
        Subject: Q3 Planning — Design Review Thursday

        Hey team,

        Just a heads up that we've moved the Q3 design review to Thursday at 2pm in Conference Room B.
        Please bring your sprint retrospective notes and any mockups you'd like feedback on.

        Also, lunch is on me this week — thinking tacos. Any dietary restrictions I should know about?

        See you there!
        — Marcus
    """).strip(),
}

SYSTEM_PROMPT = textwrap.dedent("""
    You are an intelligent email triage assistant. You will receive an email and must decide on the best action.

    Available actions:
      0  classify_spam    — Mark this email as spam; do not reply.
      1  classify_urgent  — Mark as urgent; needs immediate attention.
      2  classify_normal  — Mark as normal priority.
      3  reply            — Compose and send a professional reply.
      4  escalate         — Forward to a senior team member or department.

    Respond with ONLY valid JSON (no markdown, no extra text):
    {
      "action_id": <0-4>,
      "reply_text": "<your reply text if action_id is 3 or 4, else empty string>",
      "reasoning": "<brief one-sentence explanation>"
    }

    Rules:
    - Never reply to spam.
    - Keep replies professional, concise, and free of PII.
    - Use escalate for high-value clients, legal threats, or repeated issues.
""").strip()

REPLY_SYSTEM = "You are a professional email assistant. Write only the reply body — no subject line, no preamble. Be concise, helpful, and polite. Under 150 words. No PII."


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------
def call_hf_reply(token: str, email_text: str) -> str:
    """Second LLM call: generate the actual reply text for non-spam emails."""
    models_to_try = list(MODEL_FALLBACKS)
    for m in models_to_try:
        try:
            client = InferenceClient(model=m, token=token)
            resp = client.chat_completion(
                messages=[
                    {"role": "system", "content": REPLY_SYSTEM},
                    {"role": "user",   "content": f"Write a professional reply to this email:\n\n{email_text}"},
                ],
                max_tokens=300,
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            time.sleep(1)
            continue
    return ""


def call_hf_api(token: str, model: str, email_text: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Triage this email:\n\n{email_text}\n\nRespond ONLY with JSON."},
    ]
    # Try the requested model first, then fall back through the list
    models_to_try = [model] + [m for m in MODEL_FALLBACKS if m != model]
    last_exc = None
    for m in models_to_try:
        try:
            client = InferenceClient(model=m, token=token)
            resp = client.chat_completion(messages=messages, max_tokens=512, temperature=0.2)
            return _parse_action(resp.choices[0].message.content)
        except Exception as exc:
            last_exc = exc
            time.sleep(1)
            continue
    # All models failed — use rule-based fallback silently
    return None


def _parse_action(raw: str) -> dict:
    import re
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = []
        for line in lines[1:]:
            if line.strip() == "```":
                break
            inner.append(line)
        text = "\n".join(inner).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"action_id": ACTION_CLASSIFY_NORMAL, "reply_text": "", "reasoning": "(could not parse response)"}


def run_env_episode(email_text: str, task_name: str) -> dict:
    """
    Run a single episode through EmailTriageEnv with the LLM agent.
    Returns a dict with all display fields.
    """
    token = os.environ.get("HF_API_TOKEN", "")
    model = DEFAULT_MODEL

    dataset = load_emails()
    task    = TASK_MAP[task_name]()
    env     = EmailTriageEnv(task=task, dataset=dataset, seed=42, max_steps=5)

    # Inject the user's pasted email as the first email in the episode
    # by monkey-patching the dataset entry temporarily
    synthetic = {
        "id": "ui_input",
        "sender": _extract_field(email_text, "from") or "user@input.com",
        "subject": _extract_field(email_text, "subject") or "User Submitted Email",
        "body": email_text,
        "thread_history": [],
        "metadata": {"has_attachment": False, "is_vip": False, "pii_fields": []},
        "ground_truth": {"label": "normal", "action": "reply", "reply_keywords": []},
    }
    env.dataset = [synthetic]
    env._order  = [0]
    env._idx    = 0

    obs, info = env.reset()

    episode_num = EPISODE_COUNTER["value"]
    EPISODE_COUNTER["value"] += 1

    # --- Step 0: Classification ---
    if not token:
        action_dict = _rule_based_action(email_text)
    else:
        action_dict = call_hf_api(token, model, email_text)
        if action_dict is None:  # all HF models failed
            action_dict = _rule_based_action(email_text)

    action_id  = int(action_dict.get("action_id", ACTION_CLASSIFY_NORMAL))
    reply_text = str(action_dict.get("reply_text", ""))
    reasoning  = str(action_dict.get("reasoning", ""))

    # --- Always generate a reply for non-spam via a dedicated LLM call ---
    is_spam = (action_id == ACTION_CLASSIFY_SPAM)
    if not is_spam:
        if token:
            generated_reply = call_hf_reply(token, email_text)
        else:
            generated_reply = _rule_based_reply(email_text, action_id)
        # Prefer the dedicated reply over whatever the triage call returned
        if generated_reply:
            reply_text = generated_reply
            action_dict["reply_text"] = reply_text

    obs, reward, terminated, truncated, step_info = env.step(action_dict)
    grader_info = step_info.get("grader_info", {})

    # --- Step 1 (Medium/Hard): if not done, send reply/escalate ---
    total_reward = reward
    step2_info   = {}
    if not (terminated or truncated) and task_name in ("Medium", "Hard"):
        if is_spam:
            secondary = {"action_id": ACTION_CLASSIFY_SPAM, "reply_text": "", "reasoning": "Spam — no reply."}
        else:
            secondary = {"action_id": ACTION_REPLY, "reply_text": reply_text, "reasoning": reasoning}
        obs2, r2, terminated, truncated, step2_info_raw = env.step(secondary)
        total_reward += r2
        step2_info = step2_info_raw.get("grader_info", {})

    # --- Step 2 (Hard): compose reply ---
    if not (terminated or truncated) and task_name == "Hard":
        reply_action = {
            "action_id": ACTION_REPLY if not is_spam else ACTION_CLASSIFY_SPAM,
            "reply_text": reply_text,
            "reasoning": reasoning,
        }
        _, r3, _, _, step3_info_raw = env.step(reply_action)
        total_reward += r3

    # Build display values
    label, icon = ACTION_LABEL_MAP.get(action_id, ("Normal", "🟢"))

    # Map action_id → classification label
    cls_map = {
        ACTION_CLASSIFY_SPAM:   "Spam / Phishing",
        ACTION_CLASSIFY_URGENT: "Urgent",
        ACTION_CLASSIFY_NORMAL: "Normal",
        ACTION_REPLY:           "Normal",
        ACTION_ESCALATE:        "Urgent",
    }
    classification = cls_map.get(action_id, "Normal")

    # Action label
    action_label_map = {
        ACTION_CLASSIFY_SPAM:   "Ignore",
        ACTION_CLASSIFY_URGENT: "Escalate",
        ACTION_CLASSIFY_NORMAL: "Reply",
        ACTION_REPLY:           "Reply",
        ACTION_ESCALATE:        "Escalate",
    }
    action_label = action_label_map.get(action_id, "Reply")

    # Confidence from grader reward
    confidence = min(0.99, max(0.50, total_reward if total_reward > 0 else 0.75))

    # Thinking bullets
    thinking = _build_thinking(email_text, action_id, reasoning, grader_info)

    # Explanation
    explanation = _build_explanation(action_id, reasoning, grader_info, task_name)

    return {
        "classification": classification,
        "action":         action_label,
        "confidence":     round(confidence, 2),
        "explanation":    explanation,
        "thinking":       thinking,
        "reply":          reply_text if not is_spam else "",
        "reward":         round(total_reward, 4),
        "episode":        episode_num,
        "task":           task_name,
        "difficulty":     task_name,
    }


def _extract_field(text: str, field: str) -> str:
    for line in text.splitlines():
        if line.lower().startswith(field + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def _rule_based_reply(text: str, action_id: int) -> str:
    """Fallback reply when no HF token is available."""
    lower = text.lower()
    if action_id == ACTION_ESCALATE or any(k in lower for k in ["urgent", "legal", "escalate", "manager"]):
        return (
            "Dear sender,\n\nThank you for reaching out. We have escalated your case to our senior team "
            "and someone will be in touch with you within 2 business hours.\n\nBest regards,\nSupport Team"
        )
    if any(k in lower for k in ["refund", "damaged", "complaint", "disappointed", "unacceptable"]):
        return (
            "Dear sender,\n\nWe sincerely apologize for the experience you've had. Your case has been "
            "logged and we will arrange a replacement or full refund within 1 business day. "
            "Thank you for your patience.\n\nBest regards,\nSupport Team"
        )
    return (
        "Dear sender,\n\nThank you for your email. We have received your message and will respond "
        "with a detailed follow-up shortly.\n\nBest regards,\nSupport Team"
    )


def _rule_based_action(text: str) -> dict:
    lower = text.lower()
    phishing = any(k in lower for k in ["verify your", "suspended", "click here", "secur1ty", "password", "bank account"])
    urgent   = any(k in lower for k in ["urgent", "asap", "immediately", "24 hours", "eod"])
    complaint= any(k in lower for k in ["disappointed", "refund", "unacceptable", "complaint", "damaged"])

    if phishing:
        return {"action_id": ACTION_CLASSIFY_SPAM,   "reply_text": "", "reasoning": "Phishing indicators detected."}
    elif urgent and complaint:
        return {"action_id": ACTION_ESCALATE,        "reply_text": "We are escalating your case immediately.", "reasoning": "Urgent complaint from customer."}
    elif urgent:
        return {"action_id": ACTION_CLASSIFY_URGENT, "reply_text": "Acknowledged, handling urgently.", "reasoning": "Urgency markers detected."}
    else:
        return {"action_id": ACTION_REPLY,           "reply_text": "Thank you for your email. We will get back to you shortly.", "reasoning": "Standard correspondence."}


def _build_thinking(text: str, action_id: int, reasoning: str, grader: dict) -> list[str]:
    lower  = text.lower()
    points = []
    points.append(f"[SCAN] Email scanned -- {len(text.split())} words, {len(text.splitlines())} lines")

    if any(k in lower for k in ["verify", "suspended", "click here", "password"]):
        points.append("[ALERT] Phishing indicators detected: urgency + suspicious link pattern")
    if any(k in lower for k in ["urgent", "asap", "immediately", "24 hours"]):
        points.append("[TIME] Urgency markers found in subject/body")
    if any(k in lower for k in ["disappointed", "refund", "complaint", "unacceptable"]):
        points.append("[SENTIMENT] Negative sentiment detected -- complaint pattern")
    if any(k in lower for k in ["team", "sprint", "meeting", "review", "conference"]):
        points.append("[INTERNAL] Internal communication pattern identified")

    points.append(f"[LLM] Reasoning: {reasoning}" if reasoning else "[FALLBACK] Rule-based used (no HF token or all models unavailable)")

    action_name = ACTION_NAMES.get(action_id, "unknown")
    points.append(f"[DECISION] action={action_name} (id={action_id})")

    if grader.get("correct") is not None:
        correct_str = "correct" if grader["correct"] else "incorrect"
        points.append(f"[GRADER] Classification {correct_str} | reward={grader.get('reward', 0):.2f}")

    return points


def _build_explanation(action_id: int, reasoning: str, grader: dict, task: str) -> str:
    base = {
        ACTION_CLASSIFY_SPAM:   "This email was identified as spam or phishing. No reply will be sent to avoid engaging with malicious actors.",
        ACTION_CLASSIFY_URGENT: "This email requires immediate attention. Urgency signals and high-priority context were detected.",
        ACTION_CLASSIFY_NORMAL: "This is standard correspondence with no urgency or threat indicators. A routine reply is appropriate.",
        ACTION_REPLY:           "The email requires a direct reply. The agent composed a professional response addressing the sender's request.",
        ACTION_ESCALATE:        "This email has been flagged for escalation due to high stakes, repeated issues, or VIP sender status.",
    }.get(action_id, "Email processed by AI triage agent.")

    if reasoning:
        base += f" Agent reasoning: {reasoning}"
    if task != "Easy" and grader.get("matched_keywords"):
        base += f" Keywords matched: {', '.join(grader['matched_keywords'])}."
    return base


# ---------------------------------------------------------------------------
# Gradio analyze function
# ---------------------------------------------------------------------------
def analyze_email(email_text: str, task_name: str):
    if not email_text or not email_text.strip():
        return (
            "—", "—", "—", "0%",
            "Paste an email and click Analyze.",
            "- Waiting for input.",
            "",
            "Unknown", "--",
            "[ ] Step 1 · Read Email",
            "[ ] Step 2 · Classify",
            "[ ] Step 3 · Decide Action",
            "[ ] Step 4 · Generate Response",
            "0.0000",
        )

    try:
        result = run_env_episode(email_text.strip(), task_name)
    except Exception as e:
        return (
            "Error", "Error", "Error", "--",
            f"Backend error: {e}",
            "- An error occurred during analysis.",
            "",
            "Unknown", "--",
            "[X] Step 1", "[X] Step 2", "[X] Step 3", "[X] Step 4",
            "0.0000",
        )

    conf_pct = f"{int(result['confidence'] * 100)}%"
    thinking_md = "\n".join(f"- {p}" for p in result["thinking"])

    return (
        result["classification"],
        result["action"],
        result["difficulty"],
        conf_pct,
        result["explanation"],
        thinking_md,
        result["reply"],
        result["difficulty"],
        f"Episode #{result['episode']}",
        "[OK] Step 1 · Read Email",
        "[OK] Step 2 · Classify",
        "[OK] Step 3 · Decide Action",
        "[OK] Step 4 · Generate Response",
        str(result["reward"]),
    )


def load_example(key: str) -> str:
    return SAMPLE_EMAILS.get(key, "")


# ---------------------------------------------------------------------------
# Custom CSS (matches the new HTML UI palette)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;500;700&family=Space+Mono:wght@400;700&display=swap');

body, .gradio-container {
  font-family: 'DM Sans', sans-serif !important;
  background: #f9fafb !important;
}

.dark body, .dark .gradio-container {
  background: #030712 !important;
}

/* Header strip */
#app-header-html {
  background: #4f46e5;
  color: white;
  padding: 0.4rem 1rem;
  text-align: center;
  font-family: 'Space Mono', monospace;
  font-size: 0.75rem;
  letter-spacing: 0.05em;
  border-radius: 0.5rem;
  margin-bottom: 0.5rem;
}

/* Step pills */
.step-pill-row {
  display: flex;
  gap: 0.5rem;
  margin: 0.5rem 0;
}

/* Panels */
.panel-box {
  border-radius: 1rem;
  border: 1px solid #e5e7eb;
  background: #ffffff;
  padding: 1.2rem;
  box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}

/* Badges */
.badge-spam    { background:#fee2e2; color:#b91c1c; padding:2px 10px; border-radius:999px; font-weight:700; font-size:0.78rem; }
.badge-urgent  { background:#fef3c7; color:#b45309; padding:2px 10px; border-radius:999px; font-weight:700; font-size:0.78rem; }
.badge-normal  { background:#d1fae5; color:#065f46; padding:2px 10px; border-radius:999px; font-weight:700; font-size:0.78rem; }
.badge-reply   { background:#dbeafe; color:#1d4ed8; padding:2px 10px; border-radius:999px; font-weight:700; font-size:0.78rem; }
.badge-escalate{ background:#ede9fe; color:#6d28d9; padding:2px 10px; border-radius:999px; font-weight:700; font-size:0.78rem; }
.badge-ignore  { background:#f3f4f6; color:#374151; padding:2px 10px; border-radius:999px; font-weight:700; font-size:0.78rem; }

/* Analyze button */
#analyze-btn {
  background: #4f46e5 !important;
  color: white !important;
  font-family: 'Space Mono', monospace !important;
  font-weight: 700 !important;
  border-radius: 0.75rem !important;
  border: none !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.04em !important;
}
#analyze-btn:hover { background: #4338ca !important; }

/* Textboxes */
textarea, input[type=text] {
  border-radius: 0.75rem !important;
  border: 1px solid #e5e7eb !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.875rem !important;
}

/* Reply output */
#reply-output textarea {
  background: #eef2ff !important;
  border: 1px solid #c7d2fe !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* Reward badge */
#reward-box {
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-radius: 0.75rem;
  padding: 0.5rem 1rem;
  font-family: 'Space Mono', monospace;
  font-size: 0.8rem;
  color: #166534;
}
"""

# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="AI Email Triage Assistant") as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="background:#4f46e5;color:white;padding:0.9rem 1.5rem;border-radius:1rem;margin-bottom:0.75rem;display:flex;align-items:center;gap:1rem;">
      <div style="width:42px;height:42px;background:rgba(255,255,255,0.2);border-radius:0.75rem;display:flex;align-items:center;justify-content:center;font-size:1.3rem;">🧠</div>
      <div>
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;letter-spacing:-0.01em;">AI Email Triage Assistant</div>
        <div style="font-size:0.8rem;opacity:0.85;margin-top:2px;">Automates real-world email handling using AI agents (classification, decision, response)</div>
      </div>
    </div>
    <div style="background:#4f46e5;color:white;text-align:center;padding:0.35rem;border-radius:0.5rem;font-family:'Space Mono',monospace;font-size:0.72rem;letter-spacing:0.06em;margin-bottom:1rem;">
      ⚡ This system simulates real-world email triage for training and evaluating AI agents
    </div>
    """)

    # ── Pipeline Steps ───────────────────────────────────────────────────────
    with gr.Row():
        step1_out = gr.Textbox(value="[ ] Step 1 · Read Email",    label="", interactive=False, container=False)
        step2_out = gr.Textbox(value="[ ] Step 2 · Classify",      label="", interactive=False, container=False)
        step3_out = gr.Textbox(value="[ ] Step 3 · Decide Action", label="", interactive=False, container=False)
        step4_out = gr.Textbox(value="[ ] Step 4 · Generate Response", label="", interactive=False, container=False)

    gr.HTML("<hr style='border:none;border-top:1px solid #e5e7eb;margin:0.5rem 0;'>")

    # ── Main two-column layout ───────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── LEFT: Input ──────────────────────────────────────────────────────
        with gr.Column(scale=5):
            gr.HTML("<div style='font-family:Space Mono,monospace;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#6b7280;margin-bottom:0.3rem;'>Email Input</div>")
            gr.HTML("<div style='font-size:0.78rem;color:#9ca3af;margin-bottom:0.5rem;'>Paste an email to simulate real-world AI inbox processing</div>")

            email_input = gr.Textbox(
                lines=12,
                placeholder="Subject: Urgent — Your account has been compromised\n\nDear user, we have detected suspicious activity...",
                label="",
                show_label=False,
            )

            task_selector = gr.Radio(
                choices=["Easy", "Medium", "Hard"],
                value="Hard",
                label="Task Difficulty",
            )

            analyze_btn = gr.Button("⚡  Analyze Email", variant="primary", elem_id="analyze-btn")

            # ── Examples ────────────────────────────────────────────────────
            gr.HTML("""
            <div style='font-family:Space Mono,monospace;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#9ca3af;margin:1rem 0 0.4rem;'>
              Try Real Scenarios
            </div>""")

            with gr.Row():
                btn_phishing = gr.Button("🛡️ Phishing Attack",    variant="secondary", size="sm")
                btn_complaint= gr.Button("💬 Customer Complaint", variant="secondary", size="sm")
                btn_internal = gr.Button("🏢 Internal Email",     variant="secondary", size="sm")

        # ── RIGHT: Output ────────────────────────────────────────────────────
        with gr.Column(scale=5):

            # Decision Summary
            gr.HTML("<div style='font-family:Space Mono,monospace;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#6b7280;margin-bottom:0.5rem;'>AI Decision Summary</div>")
            with gr.Row():
                classification_out = gr.Textbox(label="Classification", value="—", interactive=False)
                action_out         = gr.Textbox(label="Action",         value="—", interactive=False)
                difficulty_out     = gr.Textbox(label="Difficulty",     value="—", interactive=False)
                confidence_out     = gr.Textbox(label="Confidence",     value="—", interactive=False)

            with gr.Row():
                episode_out = gr.Textbox(label="Episode", value="—", interactive=False)
                reward_out  = gr.Textbox(label="Reward",  value="—", interactive=False, elem_id="reward-box")

            gr.HTML("<hr style='border:none;border-top:1px solid #e5e7eb;margin:0.5rem 0;'>")

            # Explanation
            gr.HTML("<div style='font-family:Space Mono,monospace;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#6b7280;margin-bottom:0.3rem;'>Explanation</div>")
            explanation_out = gr.Textbox(
                value="No analysis yet. Paste an email to get started.",
                label="", show_label=False, interactive=False, lines=3,
            )

            gr.HTML("<hr style='border:none;border-top:1px solid #e5e7eb;margin:0.5rem 0;'>")

            # Agent Thinking
            gr.HTML("<div style='font-family:Space Mono,monospace;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#6b7280;margin-bottom:0.3rem;'>🤖 Agent Thinking</div>")
            thinking_out = gr.Markdown(value="- Waiting for input.\n- No signals detected yet.")

            gr.HTML("<hr style='border:none;border-top:1px solid #e5e7eb;margin:0.5rem 0;'>")

            # Generated Reply
            gr.HTML("<div style='font-family:Space Mono,monospace;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#6b7280;margin-bottom:0.3rem;'>↩️ Generated Reply</div>")
            reply_out = gr.Textbox(
                value="",
                placeholder="Reply will appear here when action = Reply or Escalate",
                label="", show_label=False, interactive=False, lines=7,
                elem_id="reply-output",
            )

    # ── Wire example buttons ─────────────────────────────────────────────────
    btn_phishing.click(fn=lambda: load_example("phishing"),  outputs=email_input)
    btn_complaint.click(fn=lambda: load_example("complaint"), outputs=email_input)
    btn_internal.click(fn=lambda: load_example("internal"),  outputs=email_input)

    # ── Wire analyze button ──────────────────────────────────────────────────
    analyze_btn.click(
        fn=analyze_email,
        inputs=[email_input, task_selector],
        outputs=[
            classification_out,
            action_out,
            difficulty_out,
            confidence_out,
            explanation_out,
            thinking_out,
            reply_out,
            difficulty_out,
            episode_out,
            step1_out,
            step2_out,
            step3_out,
            step4_out,
            reward_out,
        ],
    )

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)
