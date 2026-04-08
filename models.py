"""
Pydantic models for OpenEnv compliance.

Defines the Observation, Action, and Reward data types
used throughout the environment.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ObservationModel(BaseModel):
    """
    Observation model for the Email Triage environment.
    
    All observations include the current email details and step counter.
    """
    email_id: str = Field(..., description="Unique email identifier", max_length=64)
    sender: str = Field(..., description="Email sender address", max_length=256)
    subject: str = Field(..., description="Email subject line", max_length=512)
    body: str = Field(..., description="Email body text", max_length=8192)
    thread_history: str = Field(
        ..., 
        description="JSON-serialized list of previous messages in thread",
        max_length=16384
    )
    metadata: str = Field(
        ...,
        description="JSON-serialized dict with is_vip, pii_fields, etc.",
        max_length=1024
    )
    step_count: int = Field(..., description="Current step number in episode", ge=0, le=100)

    class Config:
        json_schema_extra = {
            "example": {
                "email_id": "email_001",
                "sender": "customer@example.com",
                "subject": "Account access issue",
                "body": "I cannot log into my account...",
                "thread_history": "[]",
                "metadata": "{}",
                "step_count": 0,
            }
        }


class ActionModel(BaseModel):
    """
    Action model for the Email Triage environment.
    
    The agent chooses an action_id (0-4) and optionally provides reply text.
    
    Action IDs:
      0 - classify_spam
      1 - classify_urgent
      2 - classify_normal
      3 - reply
      4 - escalate
    """
    action_id: int = Field(..., description="Action ID (0-4)", ge=0, le=4)
    reply_text: str = Field(
        default="",
        description="Reply text for reply/escalate actions",
        max_length=4096
    )

    class Config:
        json_schema_extra = {
            "example": {
                "action_id": 2,
                "reply_text": "",
            }
        }


class RewardModel(BaseModel):
    """
    Reward model for the Email Triage environment.
    
    Provides feedback on the quality of the agent's action.
    """
    reward: float = Field(
        ...,
        description="Reward signal in range [0.0, 1.0]",
        ge=0.0,
        le=1.0
    )
    done: bool = Field(..., description="Whether the episode has terminated")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional information from grader (grading details, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "reward": 0.75,
                "done": False,
                "info": {
                    "grader_info": {
                        "classification_correct": True,
                        "reasoning": "Correct classification"
                    }
                },
            }
        }


class StepResponseModel(BaseModel):
    """
    Response from a step() call.
    
    Contains the new observation, reward, and episode status.
    """
    observation: ObservationModel
    reward: float = Field(..., description="Reward for this step", ge=0.0, le=1.0)
    done: bool = Field(..., description="Whether episode has terminated")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional information"
    )


class ResetResponseModel(BaseModel):
    """
    Response from a reset() call.
    
    Contains the initial observation for a new episode.
    """
    observation: ObservationModel
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional information (task, difficulty, etc.)"
    )


class StateModel(BaseModel):
    """
    Current state of the environment.
    
    Includes both observation and internal state information.
    """
    observation: ObservationModel
    episode_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the current episode"
    )
