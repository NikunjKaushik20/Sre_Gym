"""Type-safe Pydantic models for SRE-Gym environment."""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


# ── Sub-models (not OpenEnv base classes, just Pydantic) ──

class Alert(BaseModel):
    """A monitoring alert fired by a service."""
    id: str
    service: str
    severity: str = Field(description="low | medium | high | critical")
    timestamp: int
    message: str


class LogEntry(BaseModel):
    """A single log line from a service."""
    t: int
    service: str
    msg: str
    level: str = Field(default="error", description="info | warn | error | fatal")


class IncidentState(BaseModel):
    """Current state of the incident visible to the agent."""
    acknowledged: bool = False
    diagnosed: bool = False
    diagnosis_service: Optional[str] = None
    remediated: bool = False
    escalated: bool = False
    resolved: bool = False
    step: int = 0
    max_steps: int = 15


# ── OpenEnv Action ──

class SREAction(Action):
    """Structured action for SRE incident response.

    action_type must be one of:
      - query_logs: Get logs for a service (payload: {service, window})
      - query_metrics: Get metrics for a service (payload: {service})
      - submit_diagnosis: Identify root cause (payload: {suspected_service, suspected_cause})
      - apply_remediation: Apply a fix (payload: {playbook_id})
      - escalate: Escalate incident (payload: {severity, team})
      - submit_postmortem: Close with analysis (payload: {root_cause, affected_services, timeline_steps, prevention_steps})
    """
    action_type: str = Field(
        description="Action type: query_logs | query_metrics | submit_diagnosis | "
                    "apply_remediation | escalate | submit_postmortem"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters"
    )


# ── OpenEnv Observation ──

class SREObservation(Observation):
    """Full observation returned to the agent after each step.

    Inherits `done` (bool) and `reward` (float|None) from base Observation.
    """
    alerts: List[Alert] = Field(default_factory=list)
    logs: List[LogEntry] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict)
    incident_state: IncidentState = Field(default_factory=IncidentState)
    available_playbooks: List[str] = Field(default_factory=list)
    message: str = ""


# ── OpenEnv State ──

class SREState(State):
    """Internal environment state (extends base State with episode_id, step_count)."""
    scenario_id: str = ""
    task_id: str = ""
    difficulty: str = ""
    acknowledged: bool = False
    diagnosed: bool = False
    diagnosed_service: Optional[str] = None
    diagnosed_cause: Optional[str] = None
    remediated: bool = False
    remediation_applied: Optional[str] = None
    escalated: bool = False
    escalation_team: Optional[str] = None
    resolved: bool = False
    cumulative_reward: float = 0.0
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
