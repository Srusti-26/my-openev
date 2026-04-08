"""
HardTask — Multi-step reasoning with full thread context.

Episode structure (up to 4 steps):
  Step 0 : Classify the email (spam/urgent/normal) — 20 % weight
  Step 1 : Decide secondary action:
             - reply     → continue to step 2
             - escalate  → continue to step 2 (escalation memo)
             - classify  → penalised (should have done this at step 0)
           Weight: 20 %
  Step 2 : Compose reply OR escalation memo — 60 % weight

The agent must use thread_history to detect:
  - Reply-all storms (penalise reply)
  - Repeated unresolved issues (reward escalate)
  - VIP senders in metadata (reward urgency + reply)

Reward shaping design:
  correct_cls          → 0.20
  correct_action_type  → 0.20
  reply_quality        → up to 0.60
    - keyword match    → 0.30
    - safety check     → 0.15  (no PII leakage, no promises beyond scope)
    - tone check       → 0.15

Penalties:
  -0.15 → replying to spam
  -0.10 → leaking PII keywords in reply
  -0.05 → incorrect action type at step 1
"""

from __future__ import annotations

from env.email_env import (
    ACTION_CLASSIFY_NORMAL,
    ACTION_CLASSIFY_SPAM,
    ACTION_CLASSIFY_URGENT,
    ACTION_ESCALATE,
    ACTION_REPLY,
    EmailState,
)
from graders.classification_grader import ClassificationGrader
from graders.reply_grader import ReplyGrader
from graders.safety_grader import SafetyGrader
from tasks.base_task import BaseTask

CLASSIFY_ACTIONS = {ACTION_CLASSIFY_SPAM, ACTION_CLASSIFY_URGENT, ACTION_CLASSIFY_NORMAL}


class HardTask(BaseTask):
    name = "hard_multi_step"
    difficulty = "hard"

    def __init__(self):
        self._cls_grader    = ClassificationGrader()
        self._reply_grader  = ReplyGrader()
        self._safety_grader = SafetyGrader()

    def on_reset(self, state: EmailState, raw_email: dict) -> EmailState:
        gt = raw_email["ground_truth"]
        state.metadata["ground_truth_label"]          = gt["label"]
        state.metadata["ground_truth_action"]         = gt.get("action", "reply")
        state.metadata["ground_truth_reply_keywords"] = gt.get("reply_keywords", [])
        state.metadata["is_vip"]                      = raw_email.get("metadata", {}).get("is_vip", False)
        state.metadata["pii_fields"]                  = raw_email.get("metadata", {}).get("pii_fields", [])
        state.metadata["thread_repeats"]              = self._detect_repeats(state.thread_history)

        # Accumulators
        state.metadata["reward_cls"]    = 0.0
        state.metadata["reward_action"] = 0.0
        return state

    # ------------------------------------------------------------------
    def evaluate(
        self,
        state: EmailState,
        action_id: int,
        reply_text: str,
    ) -> tuple[float, bool, dict]:

        step = state.step_count - 1

        # ---- Step 0: Classification (20 %) --------------------------------
        if step == 0:
            if action_id not in CLASSIFY_ACTIONS:
                return -0.05, False, {"reason": "expected_classification_at_step0"}

            gt_label = state.metadata["ground_truth_label"]
            cls_r, cls_info = self._cls_grader.score(action_id, gt_label)
            partial = round(cls_r * 0.20, 4)
            state.metadata["reward_cls"] = partial
            return partial, False, {**cls_info, "partial": partial}

        # ---- Step 1: Action decision (20 %) --------------------------------
        if step == 1:
            gt_action    = state.metadata["ground_truth_action"]
            is_spam      = state.metadata["ground_truth_label"] == "spam"
            has_repeats  = state.metadata["thread_repeats"]

            # Replying to spam is unsafe → heavy penalty, end episode
            if is_spam and action_id == ACTION_REPLY:
                return -0.15, True, {"reason": "replied_to_spam", "penalty": -0.15}

            correct_action_map = {
                "reply":    ACTION_REPLY,
                "escalate": ACTION_ESCALATE,
            }
            expected_id = correct_action_map.get(gt_action, ACTION_REPLY)
            correct     = (action_id == expected_id)

            # Bonus: agent correctly escalates a thread with repeated issues
            escalation_bonus = 0.05 if (has_repeats and action_id == ACTION_ESCALATE) else 0.0

            partial = round((0.20 if correct else 0.0) + escalation_bonus, 4)
            state.metadata["reward_action"] = partial
            state.metadata["chosen_action"] = action_id

            return partial, False, {
                "correct": correct,
                "expected_action": gt_action,
                "escalation_bonus": escalation_bonus,
                "partial": partial,
            }

        # ---- Step 2: Reply / Escalation memo (60 %) -----------------------
        if step == 2:
            keywords    = state.metadata.get("ground_truth_reply_keywords", [])
            pii_fields  = state.metadata.get("pii_fields", [])
            chosen_act  = state.metadata.get("chosen_action", ACTION_REPLY)

            if action_id not in (ACTION_REPLY, ACTION_ESCALATE):
                return 0.0, True, {"reason": "expected_reply_or_escalate_at_step2"}

            # Keyword / semantic quality (30 %)
            kw_r, kw_info = self._reply_grader.score(
                reply_text=reply_text,
                expected_keywords=keywords,
                email_body=state.body,
            )
            kw_partial = round(kw_r * 0.30, 4)

            # Safety check (15 %)
            safety_r, safety_info = self._safety_grader.score(
                reply_text=reply_text,
                pii_fields=pii_fields,
            )
            safety_partial = round(safety_r * 0.15, 4)

            # Tone check (15 %) — proxy: reply is non-empty and reasonably long
            tone_r     = self._tone_score(reply_text, state.metadata["ground_truth_label"])
            tone_partial = round(tone_r * 0.15, 4)

            step2_total = kw_partial + safety_partial + tone_partial
            total       = round(
                state.metadata["reward_cls"] +
                state.metadata["reward_action"] +
                step2_total, 4
            )

            return step2_total, True, {
                "keyword_score":  kw_info,
                "safety_score":   safety_info,
                "tone_partial":   tone_partial,
                "step2_reward":   step2_total,
                "episode_total":  total,
            }

        return 0.0, True, {"reason": "episode_exceeded_steps"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_repeats(thread_history: list[dict]) -> bool:
        """True if the same sender has sent >1 messages in this thread."""
        senders = [m.get("sender", "") for m in thread_history]
        return len(senders) != len(set(senders))

    @staticmethod
    def _tone_score(reply_text: str, label: str) -> float:
        """
        Heuristic tone check:
        - Urgent emails should have short, decisive replies.
        - Normal emails should have professional-length replies.
        - Returns 0–1.
        """
        words = len(reply_text.split())
        if label == "urgent":
            # Ideal: 20–80 words; penalise very short (<10) or very long (>200)
            if 20 <= words <= 80:
                return 1.0
            elif words < 10:
                return max(0.0, words / 10)
            else:
                return max(0.0, 1.0 - (words - 80) / 200)
        else:
            # Normal: at least 20 words shows effort
            if words >= 20:
                return 1.0
            return max(0.0, words / 20)
