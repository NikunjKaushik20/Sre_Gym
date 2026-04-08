"""Deterministic baseline for SRE-Gym simulation engine. No API keys needed."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from server.sre_environment import SREEnvironment
from models import SREAction

TASKS = [
    ("task_easy_1","Redis OOM"),("task_easy_2","Postgres Pool"),
    ("task_easy_3","Config Typo"),("task_easy_4","Disk Full"),
    ("task_medium_1","Cascade Redis"),("task_medium_2","Auth JWT"),
    ("task_medium_3","Traffic Spike"),("task_medium_4","Cert Expiry"),
    ("task_hard_1","Config Cascade"),("task_hard_2","Network Partition"),
    ("task_hard_3","Deploy Regression"),("task_hard_4","DNS Misconfig"),
]

CAUSE_KW = {
    "oom":"memory_leak","memory":"memory_leak","maxmemory":"memory_leak",
    "pool":"pool_exhaustion","connection":"pool_exhaustion",
    "typo":"config_typo","crashloop":"config_typo","env var":"config_typo",
    "disk":"disk_full","no space":"disk_full",
    "jwt":"jwt_bug","token":"jwt_bug","401":"jwt_bug",
    "traffic":"traffic_spike","queue overflow":"traffic_spike",
    "cert":"cert_expiry","tls":"cert_expiry","expired":"cert_expiry",
    "config":"bad_config_deploy","deploy":"deploy_regression",
    "dns":"dns_misconfiguration","nxdomain":"dns_misconfiguration",
    "partition":"network_partition","split brain":"network_partition",
}

def _pick_service(obs):
    alerts = obs.get("alerts", [])
    critical = [a for a in alerts if a.get("severity") in ("critical","high")]
    if not critical: critical = alerts
    if not critical: return "unknown"
    critical.sort(key=lambda a: a.get("timestamp", 0))
    return critical[0]["service"]

def _infer_cause(obs):
    text = " ".join(
        [l.get("msg","") for l in obs.get("logs",[])] +
        [a.get("message","") for a in obs.get("alerts",[])]
    ).lower()
    scores = {}
    for kw, cause in CAUSE_KW.items():
        if kw in text: scores[cause] = scores.get(cause, 0) + 1
    return max(scores, key=scores.get) if scores else "unknown"

def _pick_playbook(cause, available):
    for pb in available:
        if any(k in pb for k in (cause or "").split("_")): return pb
    return available[0] if available else "unknown"

def run(task_id, name):
    env = SREEnvironment()
    obs = env.reset(task_id=task_id, seed=42)
    d = obs.model_dump()
    suspect = _pick_service(d)

    # Query logs + metrics
    for action_type in ("query_logs", "query_metrics"):
        if not d.get("done"):
            obs = env.step(SREAction(action_type=action_type, payload={"service": suspect}))
            d = obs.model_dump()

    cause = _infer_cause(d)

    # Diagnose
    if not d.get("done"):
        obs = env.step(SREAction(action_type="submit_diagnosis",
            payload={"suspected_service": suspect, "suspected_cause": cause}))
        d = obs.model_dump()

    # Remediate
    pb = _pick_playbook(cause, d.get("available_playbooks", []))
    if not d.get("done"):
        obs = env.step(SREAction(action_type="apply_remediation", payload={"playbook_id": pb}))
        d = obs.model_dump()

    # Postmortem
    affected = list({a["service"] for a in d.get("alerts", [])})
    if not d.get("done"):
        obs = env.step(SREAction(action_type="submit_postmortem", payload={
            "root_cause": cause, "affected_services": affected,
            "timeline_steps": 4, "prevention_steps": ["monitoring", "canary_deploy", "load_testing"],
        }))
        d = obs.model_dump()

    health = d.get("metadata", {}).get("system_health", 0)
    reward = d.get("metadata", {}).get("cumulative_reward", 0)
    print(f"  {task_id:20s} {name:20s} health={health:.3f}  reward={reward:.3f}")
    return reward

def main():
    print("=" * 70)
    print("  SRE-GYM DETERMINISTIC BASELINE (no API keys)")
    print("=" * 70)
    scores = []
    for tid, name in TASKS:
        scores.append(run(tid, name))
    avg = sum(scores) / len(scores)
    print(f"\n  {'Average':42s} reward={avg:.3f}")
    print(f"  {'Tasks > 0.3':42s} {sum(1 for s in scores if s > 0.3)}/{len(scores)}")

if __name__ == "__main__":
    main()
