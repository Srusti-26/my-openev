"""
server.py — FastAPI HTTP server implementing OpenEnv API.

This server provides REST endpoints for:
- POST /reset — Initialize a new episode
- POST /step  — Submit an action and get the new observation & reward
- GET /state  — Get current environment state
- POST /seed  — Set random seed
- GET /info   — Get environment metadata

Environment variable required:
  HF_API_TOKEN (optional for server, required for inference)
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import load_emails
from env.email_env import EmailTriageEnv
from models import (
    ObservationModel,
    ActionModel,
    StepResponseModel,
    ResetResponseModel,
    StateModel,
)
from tasks.easy_task import EasyTask
from tasks.medium_task import MediumTask
from tasks.hard_task import HardTask

# ============================================================================
# Global state
# ============================================================================

TASK_MAP = {
    "easy": EasyTask,
    "medium": MediumTask,
    "hard": HardTask,
}

# Will be initialized on startup
env: EmailTriageEnv | None = None
current_task: str = "easy"
dataset: list[dict] = []

# Eagerly load dataset on module import  
try:
    dataset = load_emails()
except Exception as e:
    print(f"Warning: Failed to load dataset on import: {e}", flush=True)
    dataset = []


# ============================================================================
# Pydantic request/response models
# ============================================================================

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    action: ActionModel


class SeedsRequest(BaseModel):
    seed: int


class EnvInfoResponse(BaseModel):
    name: str = "email_triage_env"
    version: str = "1.0.0"
    description: str = "Multi-difficulty RL environment for email triage"
    tasks: list[str] = ["easy", "medium", "hard"]
    observation_fields: Dict[str, str] = {
        "email_id": "str",
        "sender": "str",
        "subject": "str",
        "body": "str",
        "thread_history": "str (JSON)",
        "metadata": "str (JSON)",
        "step_count": "int",
    }
    action_space: Dict[str, Any] = {
        "action_id": {"type": "Discrete", "n": 5, "labels": {
            "0": "classify_spam",
            "1": "classify_urgent",
            "2": "classify_normal",
            "3": "reply",
            "4": "escalate",
        }},
        "reply_text": {"type": "Text", "max_length": 4096},
    }


# ============================================================================
# Lifespan management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global env, dataset
    
    # Startup: Load dataset
    dataset = load_emails()
    print(f"Loaded {len(dataset)} emails", flush=True)
    
    yield
    
    # Cleanup (if needed)
    env = None


# ============================================================================
# FastAPI app
# ============================================================================

app = FastAPI(
    title="EmailTriageEnv OpenEnv API",
    description="OpenEnv-compatible HTTP API for email triage RL environment",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Hugging Face Spaces and other deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health check
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "dataset_loaded": len(dataset) > 0,
        "env_initialized": env is not None,
    }


# ============================================================================
# Info endpoints
# ============================================================================

@app.get("/info")
async def get_info() -> EnvInfoResponse:
    """Get environment metadata."""
    return EnvInfoResponse()


# ============================================================================
# OpenEnv API endpoints
# ============================================================================

@app.post("/reset")
async def reset(request: ResetRequest = None) -> Dict[str, Any]:
    """
    Initialize a new episode.
    
    Parameters:
    - task: "easy", "medium", or "hard"
    - seed: Random seed (optional)
    - options: Additional options (optional)
    
    Returns:
    - observation: Initial observation dict
    - info: Episode metadata
    """
    global env, current_task
    
    if request is None:
        request = ResetRequest()
    
    if request.task not in TASK_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {request.task}. Choose from {list(TASK_MAP.keys())}"
        )
    
    # Create new environment with requested task
    task = TASK_MAP[request.task]()
    env = EmailTriageEnv(
        task=task,
        dataset=dataset,
        seed=request.seed or 42,
        max_steps=5,
        render_mode="json",
    )
    current_task = request.task
    
    obs_dict, info_dict = env.reset(seed=request.seed)
    
    # Validate observation with model (side effect: ensures correct types)
    ObservationModel(**obs_dict)
    
    return {
        "observation": obs_dict,
        "info": info_dict,
    }


@app.post("/step")
async def step(request: StepRequest) -> Dict[str, Any]:
    """
    Execute one step in the environment.
    
    Parameters:
    - action: ActionModel with action_id and optional reply_text
    
    Returns:
    - observation: New observation
    - reward: Reward for this step (0.0 - 1.0)
    - done: Whether episode has terminated
    - info: Step information (grader details, etc.)
    """
    global env
    
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    # Execute step in environment
    obs_dict, reward, terminated, truncated, info = env.step({
        "action_id": request.action.action_id,
        "reply_text": request.action.reply_text,
    })
    
    done = terminated or truncated
    
    # Validate observation with model (side effect: ensures correct types)
    ObservationModel(**obs_dict)
    
    return {
        "observation": obs_dict,
        "reward": float(reward),
        "done": done,
        "info": info,
    }


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """
    Get current environment state.
    
    Returns:
    - observation: Current observation
    - episode_info: Metadata about current episode
    """
    global env
    
    if env is None or env._state is None:
        raise HTTPException(
            status_code=400,
            detail="No active environment or episode. Call /reset first."
        )
    
    obs_dict = env._state.to_obs()
    ObservationModel(**obs_dict)  # Validate
    
    episode_info = {
        "task": env.task.name,
        "difficulty": env.task.difficulty,
        "step_count": env._state.step_count,
        "done": env._state.done,
        "actions_taken": env._state.actions_taken,
    }
    
    return {
        "observation": obs_dict,
        "episode_info": episode_info,
    }


@app.post("/seed")
async def set_seed(request: SeedsRequest):
    """
    Set the random seed for reproducibility.
    
    Parameters:
    - seed: Random seed value
    """
    global env
    
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    env._rng.seed(request.seed)
    return {"seed": request.seed, "message": "Seed set successfully"}


# ============================================================================
# Utility endpoints
# ============================================================================

@app.get("/tasks")
async def list_tasks():
    """List available tasks."""
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Easy — Email Classification",
                "difficulty": "easy",
                "max_steps": 1,
                "description": "Agent must classify each email as spam, urgent, or normal.",
            },
            {
                "id": "medium",
                "name": "Medium — Classification + Reply Generation",
                "difficulty": "medium",
                "max_steps": 2,
                "description": "Two-step episode: (1) classify email, (2) generate a reply.",
            },
            {
                "id": "hard",
                "name": "Hard — Multi-step Reasoning with Context",
                "difficulty": "hard",
                "max_steps": 4,
                "description": "Multi-step reasoning: classify → decide action → craft reply with full context.",
            },
        ]
    }


# ============================================================================
# Root endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API documentation link."""
    return {
        "name": "EmailTriageEnv OpenEnv API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "GET /health": "Health check",
            "GET /info": "Environment metadata",
            "GET /tasks": "List available tasks",
            "POST /reset": "Reset environment (start new episode)",
            "POST /step": "Execute action (submit action to environment)",
            "GET /state": "Get current state",
            "POST /seed": "Set random seed",
        },
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"Starting EmailTriageEnv API server on {host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port)
