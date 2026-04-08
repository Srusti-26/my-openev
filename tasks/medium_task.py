"""
MediumTask — Classify the email AND generate an appropriate reply.

Two-step episode:
  Step 0 : Agent must classify (spam / urgent / normal).
            Reward = 0.4 for correct, 0.0 for wrong.
  Step 1 : Agent must reply (action_id=3) with a reply_text.
            Reward = up to 0.6 based on reply quality.
  Total max reward = 1.0

Design decisions:
- Splitting classification and reply across two steps forces the agent to
  build a plan rather than emit a single token.
- A wrong classification at step 0 still gives partial credit at step 1
  if the reply is semantically appropriate — this mimics real-world partial
  correctness (a human might mis-label but still write a good response).
- Spam emails have no step 1: replying to spam is unsafe, so the episode
  terminates after classification with a cap at 0.4 max reward.
"""

from __future__ import annotations

from env.email_env import (
    ACTION_CLASSIFY_NORMAL,
    ACTION_CLASSIFY_SPAM,
    ACTION_CLASSIFY_URGENT,
    ACTION_REPLY,
    EmailState,
)
from graders.classification_grader import ClassificationGrader
from graders.reply_grader import ReplyGrader
from tasks.base_task import BaseTask

CLASSIFY_ACTIONS = {ACTION_CLASSIFY_SPAM, ACTION_CLASSIFY_URGENT, ACTION_CLASSIFY_NORMAL}


class MediumTask(BaseTask):
    name = "medium_classify_reply"
    difficulty = "medium"

    def __init__(self):
        self._cls_grader   = ClassificationGrader()
        self._reply_grader = ReplyGrader()

    def on_reset(self, state: EmailState, raw_email: dict) -> EmailState:
        gt = raw_email["ground_truth"]
        state.metadata["ground_truth_label"]        = gt["label"]
        state.metadata["ground_truth_reply_keywords"] = gt.get("reply_keywords", [])
        state.metadata["cls_reward_earned"]         = 0.0
        state.metadata["cls_correct"]               = False
        return state

    def evaluate(
        self,
        state: EmailState,
        action_id: int,
        reply_text: str,
    ) -> tuple[float, bool, dict]:

        step = state.step_count - 1   # already incremented by env

        # ---- Step 0: Classification ----------------------------------------
        if step == 0:
            if action_id not in CLASSIFY_ACTIONS:
                # Wrong action type at this step → penalise and end
                return -0.1, True, {"reason": "expected_classification"}

            gt_label = state.metadata["ground_truth_label"]
            cls_reward, cls_info = self._cls_grader.score(action_id, gt_label)

            # Scale classification reward to 40 % of total
            partial = round(cls_reward * 0.4, 4)
            state.metadata["cls_reward_earned"] = partial
            state.metadata["cls_correct"]       = cls_info["correct"]

            # Spam emails: replying is unsafe — episode ends here
            if gt_label == "spam":
                done = True
                info = {**cls_info, "partial_reward": partial,
                        "note": "spam email — no reply step"}
            else:
                done = False   # continue to reply step
                info = {**cls_info, "partial_reward": partial}

            return partial, done, info

        # ---- Step 1: Reply -------------------------------------------------
        if step == 1:
            if action_id != ACTION_REPLY:
                # Agent used wrong action → no reply reward
                return 0.0, True, {"reason": "expected_reply_action"}

            keywords = state.metadata.get("ground_truth_reply_keywords", [])
            reply_reward, reply_info = self._reply_grader.score(
                reply_text=reply_text,
                expected_keywords=keywords,
                email_body=state.body,
            )

            # Scale reply reward to 60 % of total
            partial = round(reply_reward * 0.6, 4)
            total   = round(state.metadata["cls_reward_earned"] + partial, 4)

            return partial, True, {**reply_info, "total_reward": total}

        # Should not reach here, but truncate gracefully
        return 0.0, True, {"reason": "unexpected_step"}
