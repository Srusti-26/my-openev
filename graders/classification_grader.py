"""
ClassificationGrader — deterministic, accuracy-based grader.

Maps action_id → label string, then compares against ground truth.
No randomness; same input always yields same score.

Scoring:
  correct   → 1.0
  incorrect → 0.0

Design note: binary reward here (no partial credit for classification)
because ambiguous partial credit for labels trains the agent to hedge.
Partial credit lives at the task level via reward weighting.
"""

from __future__ import annotations

from env.email_env import (
    ACTION_CLASSIFY_NORMAL,
    ACTION_CLASSIFY_SPAM,
    ACTION_CLASSIFY_URGENT,
)

ACTION_TO_LABEL = {
    ACTION_CLASSIFY_SPAM:   "spam",
    ACTION_CLASSIFY_URGENT: "urgent",
    ACTION_CLASSIFY_NORMAL: "normal",
}


class ClassificationGrader:
    def score(self, action_id: int, ground_truth_label: str) -> tuple[float, dict]:
        """
        Parameters
        ----------
        action_id : int
            The classification action chosen by the agent.
        ground_truth_label : str
            One of "spam", "urgent", "normal".

        Returns
        -------
        (reward, info_dict)
        """
        predicted = ACTION_TO_LABEL.get(action_id, "unknown")
        correct   = predicted == ground_truth_label
        reward    = 1.0 if correct else 0.0

        return reward, {
            "grader":      "classification",
            "predicted":   predicted,
            "ground_truth": ground_truth_label,
            "correct":     correct,
            "reward":      reward,
        }
