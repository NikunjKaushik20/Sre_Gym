"""SRE-Gym Environment — core step/reset/state logic.

Implements:
  - reset(seed, task_id): load scenario, optionally randomise service name variants
  - step(action): route action, apply dense reward shaping
  - state: return internal SREState
  - Payload validation per action type
  - Dynamic metric updates after remediation
  - close_incident bail-out action
"""
import json
import random
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

from models import (
    SREAction, SREObservation, SREState,
    Alert, LogEntry, IncidentState,
)
from server.graders import grade_easy, grade_medium, grade_hard, compute_mttr_bonus

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

# Map task IDs to scenario files
TASK_SCENARIO_MAP = {
    "task_easy_1":   "easy_redis_oom.json",
    "task_easy_2":   "easy_postgres_slow.json",
    "task_easy_3":   "easy_config_typo.json",
    "task_easy_4":   "easy_disk_full.json",
    "task_medium_1": "medium_cascade_redis.json",
    "task_medium_2": "medium_cascade_auth.json",
    "task_medium_3": "medium_traffic_spike.json",
    "task_medium_4": "medium_cert_expiry.json",
    "task_hard_1":   "hard_config_cascade.json",
    "task_hard_2":   "hard_network_partition.json",
    "task_hard_3":   "hard_deploy_regression.json",
    "task_hard_4":   "hard_multi_region.json",
}

VALID_ACTION_TYPES = {
    "query_logs", "query_metrics", "submit_diagnosis",
    "apply_remediation", "escalate", "submit_postmortem",
    "close_incident",  # explicit bail-out — penalised if incident not resolved
}

# Required payload fields per action type for validation
REQUIRED_PAYLOAD_FIELDS = {
    "query_logs":        ["service"],
    "query_metrics":     ["service"],
    "submit_diagnosis":  ["suspected_service", "suspected_cause"],
    "apply_remediation": ["playbook_id"],
    "escalate":          ["team"],
    "submit_postmortem": ["root_cause", "affected_services", "timeline_steps", "prevention_steps"],
    "close_incident":    [],
}


class SREEnvironment(Environment):
    """Deterministic SRE incident response RL environment."""

    def __init__(self):
        self._state: SREState = SREState(episode_id=str(uuid4()), step_count=0)
        self._scenario: Optional[dict] = None
        self._per_step_reward: float = 0.0
        self._live_metrics: dict = {}   # evolves after actions
        self._queried_logs: list = []   # accumulates across steps

    # ─────────────────────────── PUBLIC INTERFACE ───────────────────────────

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> Observation:
        """Load scenario and return initial observation.

        Args:
            seed: If provided, sets random seed for minor variant shuffling.
            episode_id: Optional episode ID; auto-generated if omitted.
            task_id (kwarg): Which task scenario to load (default: task_easy_1).
        """
        if seed is not None:
            random.seed(seed)

        task_id = kwargs.get("task_id", "task_easy_1")
        scenario_file = TASK_SCENARIO_MAP.get(task_id, "easy_redis_oom.json")
        scenario_path = SCENARIOS_DIR / scenario_file

        with open(scenario_path) as f:
            self._scenario = json.load(f)

        # Seed-driven metric variance (±5% jitter so each seed feels distinct)
        if seed is not None:
            self._scenario = self._apply_metric_jitter(self._scenario, seed)

        self._live_metrics = dict(self._scenario.get("initial_metrics", {}))
        self._queried_logs = []

        self._state = SREState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            scenario_id=self._scenario["scenario_id"],
            task_id=task_id,
            difficulty=self._scenario["difficulty"],
        )

        return self._build_observation(
            reward=0.0, done=False,
            message="Incident opened. Investigate the alerts, query services, diagnose the root cause, and resolve."
        )

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs) -> Observation:
        """Process one agent action and return an updated observation."""
        self._state.step_count += 1
        self._per_step_reward = -0.01  # MTTR pressure: every step costs

        # Coerce to SREAction if necessary
        if not isinstance(action, SREAction):
            try:
                action = SREAction(**action.model_dump())
            except Exception:
                return self._build_observation(
                    reward=-0.05, done=False,
                    message="Invalid action format. Expected {action_type, payload}."
                )

        act_type = action.action_type
        payload  = action.payload or {}
        self._state.actions_taken.append({"type": act_type, "payload": payload})

        # Unknown action type
        if act_type not in VALID_ACTION_TYPES:
            return self._build_observation(
                reward=-0.05, done=False,
                message=f"Unknown action_type '{act_type}'. Valid: {sorted(VALID_ACTION_TYPES)}"
            )

        # Payload validation
        missing = [f for f in REQUIRED_PAYLOAD_FIELDS.get(act_type, []) if f not in payload]
        if missing:
            return self._build_observation(
                reward=-0.03, done=False,
                message=f"Missing required payload fields for '{act_type}': {missing}"
            )

        # Route to handler
        handler = getattr(self, f"_handle_{act_type}", None)
        obs = handler(payload) if handler else self._build_observation(
            reward=-0.05, done=False, message="Internal error: no handler found."
        )

        return self._check_termination(obs, act_type, payload)

    @property
    def state(self) -> State:
        """Return the current internal environment state."""
        return self._state

    # ─────────────────────── ACTION HANDLERS ────────────────────────────────

    def _handle_query_logs(self, payload: dict) -> Observation:
        service = payload.get("service", "")
        logs = self._scenario.get("queryable_logs", {}).get(service, [])
        rh_services = [rh["service"] for rh in self._scenario.get("red_herrings", [])]

        if service == self._scenario["root_cause_service"]:
            reward = 0.02    # small positive: right investigative direction
        elif service in rh_services:
            reward = -0.02   # mild negative: chasing noise
        else:
            reward = 0.0

        new_logs = [LogEntry(**l) for l in logs]
        self._queried_logs.extend(new_logs)

        return self._build_observation(
            reward=self._per_step_reward + reward, done=False,
            message=f"Queried logs for '{service}': {len(new_logs)} entries returned.",
            extra_logs=new_logs
        )

    def _handle_query_metrics(self, payload: dict) -> Observation:
        service = payload.get("service", "")
        metrics = self._scenario.get("queryable_metrics", {}).get(service, {})
        rh_services = [rh["service"] for rh in self._scenario.get("red_herrings", [])]

        if service == self._scenario["root_cause_service"]:
            reward = 0.02
        elif service in rh_services:
            reward = -0.02
        else:
            reward = 0.0

        # Merge queryable metrics into live view
        self._live_metrics.update(metrics)

        return self._build_observation(
            reward=self._per_step_reward + reward, done=False,
            message=f"Queried metrics for '{service}': {len(metrics)} values returned.",
            extra_metrics=metrics
        )

    def _handle_submit_diagnosis(self, payload: dict) -> Observation:
        suspected = payload.get("suspected_service", "")
        cause     = payload.get("suspected_cause", "")

        # Allow re-diagnosis (overwrite previous)
        self._state.diagnosed      = True
        self._state.diagnosed_service = suspected
        self._state.diagnosed_cause   = cause

        rh_services = [rh["service"] for rh in self._scenario.get("red_herrings", [])]

        if suspected in rh_services:
            reward = -0.20
            msg = (f"Diagnosed '{suspected}' — this is a red herring! "
                   f"Check the dependency graph and alert timestamps more carefully.")
        elif suspected == self._scenario["root_cause_service"]:
            if cause == self._scenario["correct_diagnosis"]:
                reward = 0.25
                msg = f"Correct! '{suspected}' with cause '{cause}' is the root cause."
            else:
                reward = 0.10
                msg = (f"Correct service '{suspected}', but wrong cause '{cause}'. "
                       f"Expected: '{self._scenario['correct_diagnosis']}'.")
            self._state.acknowledged = True
        else:
            reward = -0.05
            msg = f"Incorrect root cause diagnosis: '{suspected}'. Keep investigating."

        return self._build_observation(
            reward=self._per_step_reward + reward, done=False, message=msg
        )

    def _handle_apply_remediation(self, payload: dict) -> Observation:
        playbook = payload.get("playbook_id", "")
        self._state.remediated          = True
        self._state.remediation_applied = playbook

        if playbook == self._scenario["correct_playbook"]:
            reward = 0.20
            msg = f"Applied '{playbook}' — correct remediation! Service recovering."
            # Dynamic metric update: reflect recovery in live metrics
            self._live_metrics = self._apply_recovery_metrics()
        else:
            reward = -0.10
            msg = (f"Applied '{playbook}' — wrong remediation. "
                   f"Available playbooks: {self._scenario.get('available_playbooks', [])}")

        return self._build_observation(
            reward=self._per_step_reward + reward, done=False, message=msg
        )

    def _handle_escalate(self, payload: dict) -> Observation:
        team          = payload.get("team", "")
        severity      = payload.get("severity", "")
        self._state.escalated       = True
        self._state.escalation_team = team
        correct_team  = self._scenario.get("correct_escalation_team")

        if correct_team and team == correct_team:
            reward = 0.05
            msg = f"Escalated to '{team}' — correct team for this incident."
        elif not correct_team:
            reward = -0.05
            msg = f"Unnecessary escalation to '{team}' — this incident does not require escalation."
        else:
            reward = -0.05
            msg = f"Escalated to '{team}' — wrong team. Expected '{correct_team}'."

        return self._build_observation(
            reward=self._per_step_reward + reward, done=False, message=msg
        )

    def _handle_submit_postmortem(self, payload: dict) -> Observation:
        """Final grading action. Grade based on difficulty, apply MTTR bonus."""
        self._state.resolved = True
        score = self._compute_grade(payload)
        final = compute_mttr_bonus(score, self._state.step_count, self._scenario["max_steps"])

        return self._build_observation(
            reward=final, done=True,
            message=f"Postmortem accepted. Base score: {round(score, 3)}, MTTR-adjusted final: {final}"
        )

    def _handle_close_incident(self, payload: dict) -> Observation:
        """Explicit bail-out action. Heavy penalty if incident not properly resolved."""
        if not self._state.resolved:
            score = self._compute_grade(payload)
            final = max(0.0, score - 0.50)  # -0.50 penalty for premature close
            return self._build_observation(
                reward=final, done=True,
                message=(f"Incident closed prematurely — partial score: {final}. "
                         f"Use 'submit_postmortem' for a full resolution.")
            )
        # Already resolved via submit_postmortem — clean close
        return self._build_observation(
            reward=0.0, done=True, message="Incident closed cleanly."
        )

    # ─────────────────────── TERMINATION & GRADING ──────────────────────────

    def _check_termination(self, obs: Observation, act_type: str, payload: dict) -> Observation:
        """Force-terminate episode if max_steps exceeded."""
        if obs.done:
            return obs
        if self._state.step_count >= self._scenario["max_steps"]:
            score = self._compute_grade(payload)
            final = max(0.0, score - 0.30)  # timeout penalty
            return self._build_observation(
                reward=final, done=True,
                message=f"Max steps ({self._scenario['max_steps']}) exceeded. Timeout score: {final}"
            )
        return obs

    def _compute_grade(self, payload: dict) -> float:
        """Route to difficulty-appropriate grader."""
        diff = self._scenario.get("difficulty", "easy")
        if diff == "easy":
            return grade_easy(self._state, self._scenario)
        elif diff == "medium":
            return grade_medium(self._state, self._scenario)
        else:
            return grade_hard(payload, self._state, self._scenario)

    # ─────────────────────── OBSERVATION BUILDER ────────────────────────────

    def _build_observation(
        self,
        reward: float,
        done: bool,
        message: str,
        extra_logs: Optional[list] = None,
        extra_metrics: Optional[dict] = None,
    ) -> SREObservation:
        s = self._scenario or {}

        alerts   = [Alert(**a) for a in s.get("initial_alerts", [])]
        base_logs = [LogEntry(**l) for l in s.get("initial_logs", [])]
        # Include all previously queried logs so agent has running context
        all_logs  = base_logs + self._queried_logs
        if extra_logs:
            # Deduplicate by (t, service, msg)
            existing_keys = {(l.t, l.service, l.msg) for l in all_logs}
            for lentry in extra_logs:
                if (lentry.t, lentry.service, lentry.msg) not in existing_keys:
                    all_logs.append(lentry)
                    existing_keys.add((lentry.t, lentry.service, lentry.msg))

        metrics = dict(self._live_metrics)
        if extra_metrics:
            metrics.update(extra_metrics)

        self._state.cumulative_reward = round(
            self._state.cumulative_reward + reward, 3
        )

        return SREObservation(
            done=done,
            reward=round(reward, 3),
            alerts=alerts,
            logs=sorted(all_logs, key=lambda l: l.t),
            metrics=metrics,
            dependency_graph=s.get("dependency_graph", {}),
            incident_state=IncidentState(
                acknowledged=self._state.acknowledged,
                diagnosed=self._state.diagnosed,
                diagnosis_service=self._state.diagnosed_service,
                remediated=self._state.remediated,
                escalated=self._state.escalated,
                resolved=self._state.resolved,
                step=self._state.step_count,
                max_steps=s.get("max_steps", 15),
            ),
            available_playbooks=s.get("available_playbooks", []),
            message=message,
            metadata={
                "cumulative_reward": self._state.cumulative_reward,
                "scenario_id":       self._state.scenario_id,
                "task_id":           self._state.task_id,
                "difficulty":        self._state.difficulty,
            },
        )

    # ─────────────────────── HELPERS ────────────────────────────────────────

    def _apply_metric_jitter(self, scenario: dict, seed: int) -> dict:
        """Apply ±5% random jitter to numeric metrics for seed-based variability."""
        rng = random.Random(seed)
        jittered = json.loads(json.dumps(scenario))  # deep copy
        for key, val in jittered.get("initial_metrics", {}).items():
            if isinstance(val, (int, float)):
                factor = 1.0 + rng.uniform(-0.05, 0.05)
                jittered["initial_metrics"][key] = round(val * factor, 3)
        return jittered

    def _apply_recovery_metrics(self) -> dict:
        """Simulate metric recovery after correct remediation is applied."""
        def _recover_dict(d: dict) -> dict:
            recovered = dict(d)
            for key in list(recovered.keys()):
                lk = key.lower()
                if "error_rate" in lk or "fail_rate" in lk or "drop_rate" in lk:
                    recovered[key] = round(recovered[key] * 0.05, 3)  # 95% drop
                elif "latency" in lk or "query_time" in lk:
                    recovered[key] = round(recovered[key] * 0.20, 3)  # 80% drop
                elif "memory_pct" in lk or "disk_pct" in lk:
                    recovered[key] = round(min(recovered[key], 40.0), 3)  # cap at 40%
                elif "restarts" in lk:
                    recovered[key] = 0
                elif "rps" in lk and recovered[key] > 1000:
                    recovered[key] = round(recovered[key] * 0.25, 3)  # traffic normalises
            return recovered

        if self._scenario and "queryable_metrics" in self._scenario:
            for svc, m_dict in self._scenario["queryable_metrics"].items():
                self._scenario["queryable_metrics"][svc] = _recover_dict(m_dict)

        return _recover_dict(self._live_metrics)
