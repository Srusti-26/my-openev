"""
Test suite for EmailTriageEnv.

Tests are deterministic — no LLM calls, no randomness.
We use rule-based agents to verify reward logic exactly.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from data.loader import load_emails
from env.email_env import (
    ACTION_CLASSIFY_NORMAL,
    ACTION_CLASSIFY_SPAM,
    ACTION_CLASSIFY_URGENT,
    ACTION_ESCALATE,
    ACTION_REPLY,
    EmailTriageEnv,
)
from graders.classification_grader import ClassificationGrader
from graders.reply_grader import ReplyGrader
from graders.safety_grader import SafetyGrader
from tasks.easy_task import EasyTask
from tasks.hard_task import HardTask
from tasks.medium_task import MediumTask


@pytest.fixture
def dataset():
    return load_emails()


@pytest.fixture
def spam_email(dataset):
    return next(e for e in dataset if e["ground_truth"]["label"] == "spam")


@pytest.fixture
def urgent_email(dataset):
    return next(e for e in dataset if e["ground_truth"]["label"] == "urgent")


@pytest.fixture
def normal_email(dataset):
    return next(e for e in dataset if e["ground_truth"]["label"] == "normal")


# ==========================================================================
# Grader unit tests
# ==========================================================================
class TestClassificationGrader:
    def setup_method(self):
        self.grader = ClassificationGrader()

    def test_correct_spam(self):
        r, info = self.grader.score(ACTION_CLASSIFY_SPAM, "spam")
        assert r == 1.0
        assert info["correct"] is True

    def test_wrong_label(self):
        r, info = self.grader.score(ACTION_CLASSIFY_NORMAL, "spam")
        assert r == 0.0
        assert info["correct"] is False

    def test_deterministic(self):
        r1, _ = self.grader.score(ACTION_CLASSIFY_URGENT, "urgent")
        r2, _ = self.grader.score(ACTION_CLASSIFY_URGENT, "urgent")
        assert r1 == r2 == 1.0


class TestReplyGrader:
    def setup_method(self):
        self.grader = ReplyGrader()

    def test_perfect_keywords(self):
        r, info = self.grader.score(
            reply_text="Thank you for reaching out. We will escalate this to our billing team and resolve your invoice dispute within 48 hours.",
            expected_keywords=["escalate", "billing team", "resolve", "48 hours"],
            email_body="Invoice #4421 dispute, $3200 charge, unresolved for weeks.",
        )
        assert r > 0.7

    def test_empty_reply(self):
        r, info = self.grader.score("", ["urgent", "acknowledge"], "Server is down.")
        assert r == 0.0

    def test_no_keywords_full_credit(self):
        r, info = self.grader.score(
            reply_text="Thank you for the update, we have noted the shipment delay and will inform the relevant teams.",
            expected_keywords=[],
            email_body="Shipment delayed by 3 days due to port congestion.",
        )
        assert r > 0.5   # no keywords → kw_score=1.0, weighted with relevance/length


class TestSafetyGrader:
    def setup_method(self):
        self.grader = SafetyGrader()

    def test_safe_reply(self):
        r, info = self.grader.score("Thank you for contacting us.", [])
        assert r == 1.0
        assert info["leaked_pii"] == []

    def test_ssn_leak(self):
        r, info = self.grader.score(
            "Your SSN 123-45-6789 has been verified.", ["SSN"]
        )
        assert r < 1.0
        assert "SSN" in info["leaked_pii"]

    def test_unsafe_phrase(self):
        r, info = self.grader.score("We guarantee a full refund.", [])
        assert r < 1.0
        assert info["unsafe_phrases"]


# ==========================================================================
# Easy task integration tests
# ==========================================================================
class TestEasyTask:
    def test_correct_classification(self, dataset):
        task = EasyTask()
        env  = EmailTriageEnv(task=task, dataset=dataset, seed=0, max_steps=5)

        for ep in range(len(dataset)):
            obs, info = env.reset()
            gt = info["ground_truth"]["label"]
            action_map = {"spam": ACTION_CLASSIFY_SPAM,
                          "urgent": ACTION_CLASSIFY_URGENT,
                          "normal": ACTION_CLASSIFY_NORMAL}
            action_id = action_map[gt]

            _, reward, terminated, _, step_info = env.step(
                {"action_id": action_id, "reply_text": ""}
            )
            assert reward == 1.0, f"Expected 1.0 for {gt}, got {reward}"
            assert terminated

    def test_wrong_classification_zero_reward(self, dataset):
        task = EasyTask()
        env  = EmailTriageEnv(task=task, dataset=dataset, seed=0)
        obs, info = env.reset()
        gt = info["ground_truth"]["label"]
        wrong = ACTION_CLASSIFY_SPAM if gt != "spam" else ACTION_CLASSIFY_URGENT
        _, reward, _, _, _ = env.step({"action_id": wrong, "reply_text": ""})
        assert reward == 0.0

    def test_out_of_scope_action_penalised(self, dataset):
        task = EasyTask()
        env  = EmailTriageEnv(task=task, dataset=dataset, seed=0)
        env.reset()
        _, reward, terminated, _, _ = env.step({"action_id": ACTION_REPLY, "reply_text": "hi"})
        assert reward < 0
        assert terminated


# ==========================================================================
# Medium task integration tests
# ==========================================================================
class TestMediumTask:
    def _run_episode(self, dataset, action_seq):
        """Run an episode with a fixed list of actions."""
        task = MediumTask()
        env  = EmailTriageEnv(task=task, dataset=dataset, seed=99, max_steps=5)
        obs, info = env.reset()
        rewards = []
        for act in action_seq:
            obs, r, terminated, truncated, info = env.step(act)
            rewards.append(r)
            if terminated or truncated:
                break
        return rewards, info

    def test_spam_terminates_after_step0(self, dataset):
        # Find a spam email
        task = MediumTask()
        env  = EmailTriageEnv(task=task, dataset=dataset, seed=0, max_steps=5)
        # Keep resetting until we land on spam
        for _ in range(len(dataset)):
            obs, info = env.reset()
            if info["ground_truth"]["label"] == "spam":
                _, r, terminated, _, _ = env.step(
                    {"action_id": ACTION_CLASSIFY_SPAM, "reply_text": ""}
                )
                assert terminated
                assert r == pytest.approx(0.4, abs=0.01)
                return
        pytest.skip("No spam email found in first N resets")

    def test_total_reward_bounded(self, dataset):
        task = MediumTask()
        env  = EmailTriageEnv(task=task, dataset=dataset, seed=1, max_steps=5)
        obs, info = env.reset()
        gt = info["ground_truth"]["label"]
        cls_act = {"spam": 0, "urgent": 1, "normal": 2}[gt]

        total = 0.0
        _, r, done, trunc, _ = env.step({"action_id": cls_act, "reply_text": ""})
        total += r
        if not (done or trunc):
            _, r, done, trunc, _ = env.step({
                "action_id": ACTION_REPLY,
                "reply_text": "Thank you for your email. We will look into this shortly.",
            })
            total += r
        assert 0.0 <= total <= 1.0


# ==========================================================================
# Hard task integration tests
# ==========================================================================
class TestHardTask:
    def test_reply_to_spam_penalised(self, dataset):
        task = HardTask()
        env  = EmailTriageEnv(task=task, dataset=dataset, seed=0, max_steps=5)
        for _ in range(len(dataset)):
            obs, info = env.reset()
            if info["ground_truth"]["label"] == "spam":
                env.step({"action_id": ACTION_CLASSIFY_SPAM, "reply_text": ""})
                _, r, terminated, _, step_info = env.step(
                    {"action_id": ACTION_REPLY, "reply_text": "Hello!"}
                )
                assert r < 0
                assert terminated
                return
        pytest.skip("No spam email reached")

    def test_full_episode_reward_range(self, dataset):
        task = HardTask()
        env  = EmailTriageEnv(task=task, dataset=dataset, seed=5, max_steps=5)
        obs, info = env.reset()
        gt    = info["ground_truth"]
        label = gt["label"]
        act   = gt.get("action", "reply")

        cls_id   = {"spam": 0, "urgent": 1, "normal": 2}[label]
        act_id   = ACTION_REPLY if act == "reply" else ACTION_ESCALATE
        keywords = gt.get("reply_keywords", [])
        reply    = " ".join(keywords) + " We will address this immediately."

        total = 0.0
        for action in [
            {"action_id": cls_id,  "reply_text": ""},
            {"action_id": act_id,  "reply_text": ""},
            {"action_id": act_id,  "reply_text": reply},
        ]:
            _, r, done, trunc, _ = env.step(action)
            total += r
            if done or trunc:
                break

        assert 0.0 <= total <= 1.05   # tiny float tolerance


# ==========================================================================
# State / reset tests
# ==========================================================================
class TestEnvState:
    def test_state_returns_copy(self, dataset):
        task = EasyTask()
        env  = EmailTriageEnv(task=task, dataset=dataset, seed=0)
        env.reset()
        s1 = env.state()
        s2 = env.state()
        assert s1 is not s2
        s1.step_count = 99
        assert s2.step_count != 99

    def test_reset_cycles_dataset(self, dataset):
        task = EasyTask()
        env  = EmailTriageEnv(task=task, dataset=dataset, seed=42, max_steps=5)
        seen = set()
        for _ in range(len(dataset)):
            obs, _ = env.reset()
            seen.add(obs["email_id"])
        assert len(seen) == len(dataset)

    def test_observation_keys(self, dataset):
        task = EasyTask()
        env  = EmailTriageEnv(task=task, dataset=dataset, seed=0)
        obs, _ = env.reset()
        expected = {"email_id", "sender", "subject", "body",
                    "thread_history", "metadata", "step_count"}
        assert set(obs.keys()) == expected
