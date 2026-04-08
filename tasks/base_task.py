"""
Base class for all tasks.

Task difficulty progression:
  Easy   → single classification decision
  Medium → classification + response generation
  Hard   → multi-step reasoning: classify → decide action → craft reply
           with full thread context and escalation logic
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.email_env import EmailState


class BaseTask(ABC):
    name: str = "base"
    difficulty: str = "base"

    @abstractmethod
    def on_reset(self, state: "EmailState", raw_email: dict) -> "EmailState":
        """Called after env.reset() — can enrich state with task-specific data."""
        ...

    @abstractmethod
    def evaluate(
        self,
        state: "EmailState",
        action_id: int,
        reply_text: str,
    ) -> tuple[float, bool, dict]:
        """
        Compute reward for the action taken.

        Returns
        -------
        reward : float in [0, 1]
        done   : bool — whether the episode should terminate
        info   : dict — grader details for logging / debugging
        """
        ...
