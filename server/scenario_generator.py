"""Procedural scenario generator for SRE-Gym.

Generates randomized incident scenarios from composable building blocks,
enabling infinite task variation and curriculum-based difficulty scaling.

Usage:
    from server.scenario_generator import generate_scenario
    scenario = generate_scenario(difficulty="medium", seed=42)
"""
import json
import random
from typing import Optional

# ── Building blocks ──

SERVICE_POOL = [
    "api-gateway", "auth-service", "user-service", "payment-service",
    "order-service", "notification-service", "search-service",
    "redis", "postgres", "mongodb", "rabbitmq", "kafka",
    "cdn-proxy", "rate-limiter", "session-store",
]

ROOT_CAUSE_TEMPLATES = {
    "memory_leak": {
        "cause": "memory_leak",
        "playbook": "restart_service",
        "alert_msg": "{service} memory usage at {pct}%. OOM imminent.",
        "log_msg": "OOM killer invoked. Process killed to reclaim memory.",
        "metric_key": "memory_pct",
        "metric_val": lambda rng: round(rng.uniform(90, 99), 1),
    },
    "config_error": {
        "cause": "bad_config_deploy",
        "playbook": "rollback_config",
        "alert_msg": "{service} config reload failed — service degraded.",
        "log_msg": "Config update applied. ERROR: missing required field.",
        "metric_key": "error_rate_pct",
        "metric_val": lambda rng: round(rng.uniform(50, 100), 1),
    },
    "disk_full": {
        "cause": "disk_full",
        "playbook": "flush_logs",
        "alert_msg": "{service} disk usage at {pct}%.",
        "log_msg": "No space left on device. Write failed.",
        "metric_key": "disk_pct",
        "metric_val": lambda rng: round(rng.uniform(95, 99.9), 1),
    },
    "traffic_spike": {
        "cause": "traffic_spike",
        "playbook": "add_rate_limiting",
        "alert_msg": "{service} RPS at {val} (10x normal).",
        "log_msg": "Connection queue overflow — dropping requests.",
        "metric_key": "rps",
        "metric_val": lambda rng: rng.randint(3000, 8000),
    },
    "cert_expiry": {
        "cause": "cert_expiry",
        "playbook": "renew_certificate",
        "alert_msg": "{service} TLS certificate expired.",
        "log_msg": "TLS handshake failed: certificate has expired.",
        "metric_key": "tls_error_rate_pct",
        "metric_val": lambda rng: round(rng.uniform(80, 100), 1),
    },
}

RED_HERRING_TEMPLATES = [
    {"symptom": "brief CPU spike", "reason": "batch job completed — unrelated"},
    {"symptom": "slow query log", "reason": "off-peak vacuum — normal"},
    {"symptom": "connection spike", "reason": "auto-scaling event — resolved"},
    {"symptom": "cache miss rate up", "reason": "cold cache after deploy — transient"},
    {"symptom": "disk I/O spike", "reason": "log rotation — scheduled"},
]

PLAYBOOK_POOL = [
    "restart_service", "rollback_config", "scale_pool", "flush_logs",
    "add_rate_limiting", "renew_certificate", "flush_cache",
    "restart_queue", "failover_primary", "drain_connections",
]

PREVENTION_POOL = [
    "add_monitoring_alert", "canary_deploy", "load_testing",
    "config_validation", "capacity_planning", "runbook_update",
    "chaos_engineering", "auto_scaling_policy", "cert_rotation_automation",
]


def _build_graph(services, root_idx, rng):
    """Build a plausible dependency graph."""
    graph = {svc: [] for svc in services}
    for i, svc in enumerate(services):
        if i == root_idx:
            continue
        possible_deps = [s for j, s in enumerate(services) if j != i and j >= root_idx]
        if possible_deps:
            n_deps = rng.randint(1, min(2, len(possible_deps)))
            graph[svc] = rng.sample(possible_deps, n_deps)
    return graph


def generate_scenario(difficulty: str = "easy", seed: Optional[int] = None) -> dict:
    """Generate a random incident scenario.

    Args:
        difficulty: 'easy', 'medium', or 'hard'
        seed: random seed for reproducibility

    Returns:
        A complete scenario dict compatible with SRE-Gym.
    """
    rng = random.Random(seed)

    # Select services
    n_services = {"easy": 3, "medium": 4, "hard": 5}[difficulty]
    services = rng.sample(SERVICE_POOL, n_services)

    # Select root cause
    root_idx = rng.randint(0, len(services) - 1)
    root_service = services[root_idx]
    cause_type = rng.choice(list(ROOT_CAUSE_TEMPLATES.keys()))
    template = ROOT_CAUSE_TEMPLATES[cause_type]

    # Build graph
    graph = _build_graph(services, root_idx, rng)

    # Red herrings
    n_rh = {"easy": 0, "medium": 1, "hard": 2}[difficulty]
    rh_candidates = [s for s in services if s != root_service]
    rh_services = rng.sample(rh_candidates, min(n_rh, len(rh_candidates)))
    rh_templates = rng.sample(RED_HERRING_TEMPLATES, min(n_rh, len(RED_HERRING_TEMPLATES)))
    red_herrings = [
        {"service": svc, "fake_symptom": t["symptom"], "reason": t["reason"]}
        for svc, t in zip(rh_services, rh_templates)
    ]

    # Timestamps
    t_base = rng.randint(1000, 50000)
    metric_val = template["metric_val"](rng)

    # Build alerts
    alerts = [
        {"id": "A1", "service": root_service, "severity": "critical",
         "timestamp": t_base, "message": template["alert_msg"].format(service=root_service, pct=metric_val, val=metric_val)},
    ]
    for i, svc in enumerate(services):
        if svc != root_service:
            alerts.append({
                "id": f"A{i+2}", "service": svc, "severity": "high",
                "timestamp": t_base + rng.randint(3, 15),
                "message": f"{svc} degraded — upstream dependency failure",
            })

    # Build logs
    logs = [
        {"t": t_base - 1, "service": root_service, "msg": template["log_msg"], "level": "error"},
        {"t": t_base, "service": root_service, "msg": f"FATAL: {root_service} service failure", "level": "fatal"},
    ]
    for svc in services:
        if svc != root_service:
            logs.append({"t": t_base + rng.randint(2, 10), "service": svc,
                         "msg": f"Upstream error from {root_service}", "level": "error"})

    # Build metrics
    metrics = {f"{root_service}_{template['metric_key']}": metric_val}
    for svc in services:
        if svc != root_service:
            metrics[f"{svc}_error_rate_pct"] = round(rng.uniform(10, 60), 1)

    # Queryable data
    q_logs = {svc: [{"t": t_base + rng.randint(-5, 5), "service": svc,
                      "msg": f"Detailed log for {svc}", "level": "info"}] for svc in services}
    q_metrics = {svc: {f"{svc}_cpu": round(rng.uniform(20, 95), 1)} for svc in services}

    # Playbooks
    correct_pb = template["playbook"]
    available = list(set([correct_pb] + rng.sample(PLAYBOOK_POOL, min(5, len(PLAYBOOK_POOL)))))

    # Prevention
    prevention = rng.sample(PREVENTION_POOL, min(3, len(PREVENTION_POOL)))

    max_steps = {"easy": 10, "medium": 15, "hard": 20}[difficulty]

    return {
        "scenario_id": f"gen_{difficulty}_{seed or 'random'}",
        "task_id": f"task_gen_{difficulty}",
        "difficulty": difficulty,
        "description": f"Generated {difficulty} scenario: {cause_type} in {root_service}",
        "root_cause_service": root_service,
        "root_cause_type": cause_type,
        "correct_diagnosis": template["cause"],
        "correct_playbook": correct_pb,
        "correct_escalation_team": "infrastructure" if difficulty == "hard" else None,
        "dependency_graph": graph,
        "initial_alerts": alerts,
        "initial_logs": logs,
        "initial_metrics": metrics,
        "queryable_logs": q_logs,
        "queryable_metrics": q_metrics,
        "red_herrings": red_herrings,
        "affected_services": services,
        "valid_prevention_steps": prevention,
        "min_timeline_steps": {"easy": 2, "medium": 3, "hard": 4}[difficulty],
        "max_steps": max_steps,
        "available_playbooks": available,
    }


if __name__ == "__main__":
    for diff in ("easy", "medium", "hard"):
        s = generate_scenario(diff, seed=42)
        print(f"\n{'='*60}")
        print(f"  Generated {diff.upper()} scenario: {s['description']}")
        print(f"  Root cause: {s['root_cause_service']} ({s['correct_diagnosis']})")
        print(f"  Services: {list(s['dependency_graph'].keys())}")
        print(f"  Red herrings: {len(s['red_herrings'])}")
        print(f"  Max steps: {s['max_steps']}")
