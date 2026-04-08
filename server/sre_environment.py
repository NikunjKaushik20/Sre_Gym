"""SRE-Gym: Real-time graph-based failure simulation.

Services have continuous health [0,1]. Faults propagate through the dependency
graph each step. Fixing the root cause triggers cascading recovery; fixing
symptoms gives only temporary relief (they re-degrade next step).
"""
import json, random
from pathlib import Path
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from models import SREAction, SREObservation, SREState, Alert, LogEntry, IncidentState
from server.graders import grade_easy, grade_medium, grade_hard, compute_mttr_bonus

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"
TASK_MAP = {
    "task_easy_1": "easy_redis_oom.json", "task_easy_2": "easy_postgres_slow.json",
    "task_easy_3": "easy_config_typo.json", "task_easy_4": "easy_disk_full.json",
    "task_medium_1": "medium_cascade_redis.json", "task_medium_2": "medium_cascade_auth.json",
    "task_medium_3": "medium_traffic_spike.json", "task_medium_4": "medium_cert_expiry.json",
    "task_hard_1": "hard_config_cascade.json", "task_hard_2": "hard_network_partition.json",
    "task_hard_3": "hard_deploy_regression.json", "task_hard_4": "hard_multi_region.json",
}
VALID_ACTIONS = {"query_logs","query_metrics","submit_diagnosis","apply_remediation","escalate","submit_postmortem","close_incident"}
REQUIRED_FIELDS = {
    "query_logs": ["service"], "query_metrics": ["service"],
    "submit_diagnosis": ["suspected_service","suspected_cause"],
    "apply_remediation": ["playbook_id"], "escalate": ["team"],
    "submit_postmortem": ["root_cause","affected_services","timeline_steps","prevention_steps"],
    "close_incident": [],
}

class SREEnvironment(Environment):
    def __init__(self):
        self._state = SREState(episode_id=str(uuid4()), step_count=0)
        self._scenario = None
        self._health = {}          # service → health float [0,1]
        self._fault_service = ""
        self._fault_fixed = False
        self._initial_avg = 0.0
        self._discovered = set()
        self._queried_logs = []
        self._live_metrics = {}

    # ── PUBLIC INTERFACE ──────────────────────────────────────────────

    def reset(self, seed=None, episode_id=None, **kw):
        if seed is not None:
            random.seed(seed)
        task_id = kw.get("task_id", "task_easy_1")
        with open(SCENARIOS_DIR / TASK_MAP.get(task_id, "easy_redis_oom.json")) as f:
            self._scenario = json.load(f)

        # Init all services healthy
        self._health = {s: 1.0 for s in self._scenario["services"]}
        self._fault_service = self._scenario["fault_service"]
        self._fault_fixed = False
        self._queried_logs = []
        self._live_metrics = {}

        # Inject fault
        severity = self._scenario.get("fault_severity", 0.1)
        if seed is not None:
            severity *= (1.0 + random.uniform(-0.05, 0.05))
        self._health[self._fault_service] = max(0.02, severity)

        # Propagate 3 cycles to establish cascade
        for _ in range(3):
            self._propagate()
        self._initial_avg = self._avg_health()

        # Discover only services with health < 0.8 (they'd be alerting)
        self._discovered = {s for s, h in self._health.items() if h < 0.8}

        self._state = SREState(
            episode_id=episode_id or str(uuid4()), step_count=0,
            scenario_id=self._scenario["scenario_id"],
            task_id=task_id, difficulty=self._scenario["difficulty"],
        )
        return self._obs(0.0, False, "Incident opened. Multiple services degraded. Investigate and resolve.")

    def step(self, action, timeout_s=None, **kw):
        self._state.step_count += 1
        # Propagate simulation each step
        self._propagate()

        if not isinstance(action, SREAction):
            try: action = SREAction(**action.model_dump())
            except: return self._obs(-0.05, False, "Invalid action format.")

        at, pl = action.action_type, action.payload or {}
        self._state.actions_taken.append({"type": at, "payload": pl})

        if at not in VALID_ACTIONS:
            return self._obs(-0.05, False, f"Unknown action '{at}'.")
        missing = [f for f in REQUIRED_FIELDS.get(at, []) if f not in pl]
        if missing:
            return self._obs(-0.03, False, f"Missing fields for '{at}': {missing}")

        handler = getattr(self, f"_do_{at}", None)
        obs = handler(pl) if handler else self._obs(-0.05, False, "No handler.")
        return self._check_term(obs, at, pl)

    @property
    def state(self):
        return self._state

    # ── SIMULATION ENGINE ─────────────────────────────────────────────

    def _propagate(self):
        graph = self._scenario["dependency_graph"]
        new_h = dict(self._health)
        for svc in self._scenario["services"]:
            deps = graph.get(svc, [])
            if not deps:
                continue
            upstream = min(self._health.get(d, 1.0) for d in deps)
            if self._fault_fixed and upstream > new_h[svc]:
                # Recovery: gradually heal toward upstream health
                new_h[svc] = min(1.0, new_h[svc] + (upstream - new_h[svc]) * 0.4)
            elif upstream < 0.9:
                # Degradation: can't be healthier than weakest upstream
                cap = upstream * 0.92
                if new_h[svc] > cap:
                    new_h[svc] = max(0.02, new_h[svc] - (new_h[svc] - cap) * 0.3)
        # Fault service keeps degrading if not fixed
        if not self._fault_fixed:
            fs = self._fault_service
            new_h[fs] = max(0.01, new_h[fs] * 0.95)
        self._health = new_h

    def _avg_health(self):
        return sum(self._health.values()) / max(1, len(self._health))

    # ── ACTION HANDLERS ───────────────────────────────────────────────

    def _do_query_logs(self, pl):
        svc = pl.get("service", "")
        if svc not in self._health:
            return self._obs(-0.03, False, f"Unknown service '{svc}'.")
        self._discover(svc)
        logs = self._scenario.get("service_logs", {}).get(svc, [])
        new_logs = [LogEntry(t=self._state.step_count, service=svc, msg=l["msg"], level=l["level"]) for l in logs]
        # Add dynamic log based on current health
        h = self._health[svc]
        if h < 0.3:
            new_logs.append(LogEntry(t=self._state.step_count, service=svc, msg=f"CRITICAL: {svc} health {h:.0%}", level="fatal"))
        elif h < 0.7:
            new_logs.append(LogEntry(t=self._state.step_count, service=svc, msg=f"DEGRADED: {svc} health {h:.0%}", level="error"))
        self._queried_logs.extend(new_logs)
        r = 0.02 if svc == self._fault_service else 0.0
        return self._obs(-0.01 + r, False, f"Logs for '{svc}': {len(new_logs)} entries. Health: {h:.0%}")

    def _do_query_metrics(self, pl):
        svc = pl.get("service", "")
        if svc not in self._health:
            return self._obs(-0.03, False, f"Unknown service '{svc}'.")
        self._discover(svc)
        h = self._health[svc]
        metrics = {
            f"{svc}_health": round(h, 3),
            f"{svc}_error_rate_pct": round((1 - h) * 100, 1),
            f"{svc}_latency_p99_ms": round(50 + (1 - h) * 5000),
            f"{svc}_cpu_pct": round(20 + (1 - h) * 70, 1),
        }
        self._live_metrics.update(metrics)
        r = 0.02 if svc == self._fault_service else 0.0
        return self._obs(-0.01 + r, False, f"Metrics for '{svc}': health={h:.1%}, error_rate={(1-h)*100:.1f}%")

    def _do_submit_diagnosis(self, pl):
        suspected_svc = pl.get("suspected_service", "")
        suspected_cause = pl.get("suspected_cause", "")
        self._state.diagnosed = True
        self._state.diagnosed_service = suspected_svc
        self._state.diagnosed_cause = suspected_cause
        rh = [r["service"] for r in self._scenario.get("red_herrings", [])]
        if suspected_svc in rh:
            return self._obs(-0.15, False, f"'{suspected_svc}' is a red herring! Look deeper at the dependency graph.")
        if suspected_svc == self._fault_service:
            self._state.acknowledged = True
            r = 0.15 if suspected_cause == self._scenario.get("fault_type","") else 0.08
            return self._obs(r, False, f"Correct root cause: '{suspected_svc}'. Apply remediation now.")
        return self._obs(-0.05, False, f"'{suspected_svc}' is not the root cause. Keep investigating.")

    def _do_apply_remediation(self, pl):
        pb = pl.get("playbook_id", "")
        effects = self._scenario.get("playbook_effects", {})
        if pb not in effects:
            return self._obs(-0.05, False, f"Unknown playbook '{pb}'. Available: {self._scenario['available_playbooks']}")
        target = effects[pb]["target"]
        heal = effects[pb]["heal_amount"]
        self._state.remediated = True
        self._state.remediation_applied = pb
        if target == self._fault_service:
            self._health[target] = min(1.0, self._health[target] + heal)
            self._fault_fixed = True
            return self._obs(0.25, False, f"Applied '{pb}' to {target} — ROOT CAUSE FIXED. System recovering.")
        else:
            old_h = self._health.get(target, 0.5)
            self._health[target] = min(1.0, old_h + heal * 0.3)
            return self._obs(-0.02, False, f"Applied '{pb}' to {target} — temporary relief only. Root cause persists.")

    def _do_escalate(self, pl):
        team = pl.get("team", "")
        self._state.escalated = True
        self._state.escalation_team = team
        correct = self._scenario.get("correct_escalation_team")
        if correct and team == correct:
            return self._obs(0.03, False, f"Escalated to '{team}' — correct team.")
        return self._obs(-0.03, False, f"Escalated to '{team}'.")

    def _do_submit_postmortem(self, pl):
        self._state.resolved = True
        score = self._grade(pl)
        final = compute_mttr_bonus(score, self._state.step_count, self._scenario["max_steps"])
        return self._obs(final, True, f"Postmortem accepted. Score: {final:.3f} (base: {score:.3f})")

    def _do_close_incident(self, pl):
        score = self._grade(pl)
        final = max(0.001, min(score * 0.5, 0.999))
        return self._obs(final, True, f"Incident closed prematurely. Score: {final:.3f}")

    # ── GRADING ───────────────────────────────────────────────────────

    def _grade(self, pl):
        health_score = self._health_recovery_score()
        d = self._scenario.get("difficulty", "easy")
        if d == "easy":
            return grade_easy(health_score, self._state, self._scenario)
        elif d == "medium":
            return grade_medium(health_score, self._state, self._scenario)
        else:
            return grade_hard(health_score, pl, self._state, self._scenario)

    def _health_recovery_score(self):
        current = self._avg_health()
        if self._initial_avg >= 1.0:
            return current
        return max(0.0, min(1.0, (current - self._initial_avg) / (1.0 - self._initial_avg)))

    def _check_term(self, obs, at, pl):
        if obs.done:
            return obs
        if self._state.step_count >= self._scenario["max_steps"]:
            score = self._grade(pl)
            final = max(0.001, min(score * 0.6, 0.999))
            return self._obs(final, True, f"Max steps exceeded. Timeout score: {final:.3f}")
        return obs

    # ── OBSERVATION BUILDER ───────────────────────────────────────────

    def _obs(self, reward, done, msg):
        s = self._scenario or {}
        # Dynamic alerts from health
        alerts = []
        for svc, h in sorted(self._health.items(), key=lambda x: x[1]):
            if svc not in self._discovered:
                continue
            if h < 0.3:
                alerts.append(Alert(id=f"A-{svc}", service=svc, severity="critical",
                    timestamp=self._state.step_count, message=f"{svc} critically degraded ({h:.0%})"))
            elif h < 0.6:
                alerts.append(Alert(id=f"A-{svc}", service=svc, severity="high",
                    timestamp=self._state.step_count, message=f"{svc} unhealthy ({h:.0%})"))
            elif h < 0.85:
                alerts.append(Alert(id=f"A-{svc}", service=svc, severity="medium",
                    timestamp=self._state.step_count, message=f"{svc} degraded ({h:.0%})"))
        # Visible graph (partial observability)
        full_g = s.get("dependency_graph", {})
        vis_g = {sv: [d for d in deps if d in self._discovered]
                 for sv, deps in full_g.items() if sv in self._discovered}
        # Aggregate metrics
        metrics = dict(self._live_metrics)
        for svc in self._discovered:
            metrics[f"{svc}_health"] = round(self._health.get(svc, 1.0), 3)
        metrics["system_avg_health"] = round(self._avg_health(), 3)

        self._state.cumulative_reward = round(self._state.cumulative_reward + reward, 3)
        return SREObservation(
            done=done, reward=round(reward, 3), alerts=alerts,
            logs=sorted(self._queried_logs, key=lambda l: l.t)[-50:],
            metrics=metrics, dependency_graph=vis_g,
            incident_state=IncidentState(
                acknowledged=self._state.acknowledged, diagnosed=self._state.diagnosed,
                diagnosis_service=self._state.diagnosed_service,
                remediated=self._state.remediated, escalated=self._state.escalated,
                resolved=self._state.resolved, step=self._state.step_count,
                max_steps=s.get("max_steps", 15)),
            available_playbooks=s.get("available_playbooks", []),
            message=msg,
            metadata={"cumulative_reward": self._state.cumulative_reward,
                      "system_health": round(self._avg_health(), 3),
                      "fault_fixed": self._fault_fixed,
                      "scenario_id": self._state.scenario_id,
                      "task_id": self._state.task_id, "difficulty": self._state.difficulty})

    def _discover(self, svc):
        self._discovered.add(svc)
        for s, deps in self._scenario.get("dependency_graph", {}).items():
            if svc in deps: self._discovered.add(s)
            if s == svc: self._discovered.update(deps)
