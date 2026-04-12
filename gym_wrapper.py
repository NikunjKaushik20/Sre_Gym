"""Gymnasium-compatible wrapper for SRE-Gym.

Converts SREEnvironment into a standard gymnasium.Env so any RL library
(stable-baselines3, RLlib, CleanRL, etc.) can train on it directly.

Observation space: Box(39,) float32
  - 8 service slots × 4 features (health, error_rate, latency_norm, cpu_norm)
  - 7 system-level scalars (avg_health, step_ratio, ack, diag, remed, resolved, fault_fixed)

Action space: Discrete(19)
  - 0..7   → query_logs(services[i])
  - 8..15  → query_metrics(services[i])
  - 16     → submit_diagnosis  (auto-picks most suspicious service + inferred cause)
  - 17     → apply_remediation (best known playbook)
  - 18     → submit_postmortem (terminal action — ends episode)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from server.sre_environment import SREEnvironment
from models import SREAction

MAX_SERVICES = 8          # pad all service vectors to this fixed width
OBS_DIM = MAX_SERVICES * 4 + 7   # = 39
ACTION_DIM = 2 * MAX_SERVICES + 3  # = 19

# Keyword → canonical cause (mirrors baseline_deterministic logic)
_CAUSE_KW = {
    "oom": "memory_leak", "memory": "memory_leak", "maxmemory": "memory_leak",
    "pool": "pool_exhaustion", "connection": "pool_exhaustion",
    "typo": "config_typo", "crashloop": "config_typo", "env var": "config_typo",
    "disk": "disk_full", "no space": "disk_full",
    "jwt": "jwt_bug", "token": "jwt_bug", "401": "jwt_bug",
    "traffic": "traffic_spike", "queue overflow": "traffic_spike",
    "cert": "cert_expiry", "tls": "cert_expiry", "expired": "cert_expiry",
    "config": "bad_config_deploy", "deploy": "deploy_regression",
    "dns": "dns_misconfiguration", "partition": "network_partition",
}


class SREGymEnv(gym.Env):
    """Gymnasium wrapper around SREEnvironment.

    Parameters
    ----------
    task_id : str
        One of task_easy_1..4, task_medium_1..4, task_hard_1..4.
    seed : int | None
        Fixed random seed for reproducible episodes.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, task_id: str = "task_easy_1", seed: int | None = None):
        super().__init__()
        self._task_id = task_id
        self._seed = seed
        self._env = SREEnvironment()

        # Runtime state — set during reset()
        self._services: list[str] = []
        self._obs_raw = None
        self._best_cause = "unknown"
        self._best_playbook = "unknown"

        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

    # ── Gymnasium interface ───────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        task_id = (options or {}).get("task_id", self._task_id)
        raw = self._env.reset(seed=seed if seed is not None else self._seed,
                              task_id=task_id)
        self._obs_raw = raw
        scenario = self._env._scenario or {}
        self._services = list(scenario.get("services", []))
        self._best_cause = scenario.get("fault_type", "unknown")
        pb_list = scenario.get("available_playbooks", [])
        self._best_playbook = pb_list[0] if pb_list else "unknown"
        return self._vectorize(raw.model_dump()), {}

    def step(self, action: int):
        action_obj = self._decode_action(int(action))
        raw = self._env.step(action_obj)
        self._obs_raw = raw
        d = raw.model_dump()
        self._update_guesses(d)
        obs = self._vectorize(d)
        reward = float(d.get("reward", 0.0))
        done = bool(d.get("done", False))
        truncated = False
        info = {
            "step": self._env._state.step_count,
            "system_health": float(d.get("metadata", {}).get("system_health", 0.0)),
            "task_id": self._task_id,
            "cumulative_reward": float(d.get("metadata", {}).get("cumulative_reward", 0.0)),
        }
        return obs, reward, done, truncated, info

    def render(self):
        if self._obs_raw:
            d = self._obs_raw.model_dump()
            h = d.get("metadata", {}).get("system_health", 0.0)
            step = d.get("incident_state", {}).get("step", 0)
            print(f"  step={step:02d}  sys_health={h:.3f}  "
                  f"best_cause={self._best_cause}  "
                  f"playbook={self._best_playbook}")

    # ── Action decoding ───────────────────────────────────────────────

    def _decode_action(self, action: int) -> SREAction:
        n = max(len(self._services), 1)

        if action < MAX_SERVICES:
            svc = self._services[min(action, n - 1)]
            return SREAction(action_type="query_logs", payload={"service": svc})

        elif action < 2 * MAX_SERVICES:
            svc = self._services[min(action - MAX_SERVICES, n - 1)]
            return SREAction(action_type="query_metrics", payload={"service": svc})

        elif action == 2 * MAX_SERVICES:           # 16 — submit_diagnosis
            return SREAction(
                action_type="submit_diagnosis",
                payload={
                    "suspected_service": self._pick_suspect(),
                    "suspected_cause": self._best_cause,
                },
            )

        elif action == 2 * MAX_SERVICES + 1:       # 17 — apply_remediation
            return SREAction(
                action_type="apply_remediation",
                payload={"playbook_id": self._best_playbook},
            )

        else:                                      # 18 — submit_postmortem
            scenario = self._env._scenario or {}
            return SREAction(
                action_type="submit_postmortem",
                payload={
                    "root_cause": self._best_cause,
                    "affected_services": list(self._services),
                    "timeline_steps": [
                        "critical alert fired",
                        "queried root service logs",
                        "metrics analysed",
                        "root cause identified via dependency graph",
                        "remediation playbook applied",
                        "service health recovered",
                    ],
                    "prevention_steps": scenario.get(
                        "valid_prevention_steps",
                        ["monitoring", "canary_deploy", "load_testing"],
                    ),
                },
            )

    # ── Observation vectorisation ─────────────────────────────────────

    def _vectorize(self, d: dict) -> np.ndarray:
        vec = np.zeros(OBS_DIM, dtype=np.float32)
        metrics = d.get("metrics", {})

        for i, svc in enumerate(self._services[:MAX_SERVICES]):
            base = i * 4
            vec[base + 0] = float(metrics.get(f"{svc}_health", 1.0))
            vec[base + 1] = float(metrics.get(f"{svc}_error_rate_pct", 0.0)) / 100.0
            # latency 50–5050 ms → 0..1
            vec[base + 2] = min(float(metrics.get(f"{svc}_latency_p99_ms", 50.0)) / 5050.0, 1.0)
            vec[base + 3] = float(metrics.get(f"{svc}_cpu_pct", 20.0)) / 100.0

        # System-level features
        base = MAX_SERVICES * 4
        state = d.get("incident_state", {})
        vec[base + 0] = float(d.get("metadata", {}).get("system_health", 1.0))
        vec[base + 1] = float(state.get("step", 0)) / max(float(state.get("max_steps", 20)), 1.0)
        vec[base + 2] = float(bool(state.get("acknowledged", False)))
        vec[base + 3] = float(bool(state.get("diagnosed", False)))
        vec[base + 4] = float(bool(state.get("remediated", False)))
        vec[base + 5] = float(bool(state.get("resolved", False)))
        vec[base + 6] = float(bool(d.get("metadata", {}).get("fault_fixed", False)))
        return vec

    # ── Internal helpers ──────────────────────────────────────────────

    def _pick_suspect(self) -> str:
        """Pick service with lowest observed health; fall back to first service."""
        if not self._obs_raw:
            return self._services[0] if self._services else "unknown"
        metrics = self._obs_raw.model_dump().get("metrics", {})
        worst_svc, worst_h = None, 1.1
        for svc in self._services:
            h = float(metrics.get(f"{svc}_health", 1.0))
            if h < worst_h:
                worst_h, worst_svc = h, svc
        return worst_svc or (self._services[0] if self._services else "unknown")

    def _update_guesses(self, d: dict):
        """Infer best cause from accumulated logs + alerts; update best playbook."""
        text = " ".join(
            [l.get("msg", "") for l in d.get("logs", [])]
            + [a.get("message", "") for a in d.get("alerts", [])]
        ).lower()
        scores: dict[str, int] = {}
        for kw, cause in _CAUSE_KW.items():
            if kw in text:
                scores[cause] = scores.get(cause, 0) + 1
        if scores:
            self._best_cause = max(scores, key=scores.get)

        # Find the playbook that targets the root service
        scenario = self._env._scenario or {}
        effects = scenario.get("playbook_effects", {})
        fault_svc = scenario.get("fault_service", "")
        for pb, fx in effects.items():
            if fx.get("target") == fault_svc:
                self._best_playbook = pb
                break


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    print("Checking SREGymEnv against gymnasium spec …")
    env = SREGymEnv("task_easy_1", seed=42)
    check_env(env, warn=True)
    print("  OK gymnasium check passed")

    obs, info = env.reset(seed=0)
    print(f"  obs shape : {obs.shape}  dtype={obs.dtype}")
    print(f"  action dim: {env.action_space.n}")

    total_reward = 0.0
    for _ in range(30):
        a = env.action_space.sample()
        obs, r, done, truncated, info = env.step(a)
        total_reward += r
        if done or truncated:
            break
    print(f"  episode reward (random): {total_reward:.4f}")
    print("gym_wrapper OK")
