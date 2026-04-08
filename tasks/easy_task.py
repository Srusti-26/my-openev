"""
EasyTask — Classify the email as spam / urgent / normal.

Reward shaping:
  1.0  → correct classification
  0.0  → wrong classification
  -0.1 → used a reply/escalate action (out-of-scope for this task level)

The episode terminates after a single classification action.
This gives the agent a clean, immediate signal with no partial credit
ambiguity — ideal for initial training / curriculum learning.
"""

from __future__ import annotations

from env.email_env import (
    ACTION_CLASSIFY_NORMAL,
    ACTION_CLASSIFY_SPAM,
    ACTION_CLASSIFY_URGENT,
    EmailState,
)
from graders.classification_grader import ClassificationGrader
from tasks.base_task import BaseTask

# Actions valid for this task (no reply / escalate)
VALID_ACTIONS = {ACTION_CLASSIFY_SPAM, ACTION_CLASSIFY_URGENT, ACTION_CLASSIFY_NORMAL}


class EasyTask(BaseTask):
    name = "easy_classify"
    difficulty = "easy"

    def __init__(self):
        self._grader = ClassificationGrader()

    def on_reset(self, state: EmailState, raw_email: dict) -> EmailState:
        # Store ground-truth label in metadata so grader can access it
        state.metadata["ground_truth_label"] = raw_email["ground_truth"]["label"]
        return state

    def evaluate(
        self,
        state: EmailState,
        action_id: int,
        reply_text: str,
    ) -> tuple[float, bool, dict]:

        # Penalise out-of-scope actions immediately and end episode
        if action_id not in VALID_ACTIONS:
            return -0.1, True, {
                "reason": "out_of_scope_action",
                "action_id": action_id,
            }

        gt_label = state.metadata["ground_truth_label"]
        reward, grader_info = self._grader.score(action_id, gt_label)

        # Episode terminates after one classification
        return reward, True, grader_info
