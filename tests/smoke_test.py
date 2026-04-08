"""Smoke test for simulation-based SRE-Gym."""
import json, sys
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "server"))
sys.path.insert(0, str(root))

from graders import grade_easy, grade_medium, grade_hard, compute_mttr_bonus, _fuzzy

SCENARIOS_DIR = root / "scenarios"
P = "\033[92mPASS\033[0m"
F = "\033[91mFAIL\033[0m"
results = []

def chk(name, ok, detail=""):
    results.append(ok)
    print(f"  [{P if ok else F}] {name}" + (f" -- {detail}" if detail else ""))

def st(**kw):
    d = dict(diagnosed=False,diagnosed_service=None,diagnosed_cause=None,
             remediated=False,remediation_applied=None,escalated=False,
             escalation_team=None,acknowledged=False)
    d.update(kw)
    return SimpleNamespace(**d)

# ── Fuzzy matching ──
print("\n=== FUZZY MATCHING ===")
chk("exact", _fuzzy("memory_leak","memory_leak"))
chk("synonym oom", _fuzzy("oom","memory_leak"))
chk("substring", _fuzzy("memory_leak","memory_leak_issue"))
chk("rejects unrelated", not _fuzzy("traffic_spike","memory_leak"))

# ── Graders ──
print("\n=== GRADERS ===")
s = json.loads((SCENARIOS_DIR/"easy_redis_oom.json").read_text())
chk("easy perfect (health=1.0)", grade_easy(1.0, st(diagnosed=True,diagnosed_service="redis",
    remediated=True,remediation_applied="restart_redis"), s) == 1.0)
chk("easy zero health", grade_easy(0.0, st(), s) == 0.0)
chk("easy partial health", 0.0 < grade_easy(0.5, st(diagnosed=True,diagnosed_service="redis"), s) < 1.0)

s = json.loads((SCENARIOS_DIR/"medium_cascade_redis.json").read_text())
chk("medium no-op=0", grade_medium(0.0, st(), s) == 0.0)
chk("medium perfect", grade_medium(1.0, st(diagnosed=True,diagnosed_service="redis",
    diagnosed_cause="memory_leak",remediated=True,remediation_applied="restart_redis"), s) == 1.0)
chk("medium red herring penalty", grade_medium(0.5, st(diagnosed=True,diagnosed_service="postgres"), s) < 0.5)

s = json.loads((SCENARIOS_DIR/"hard_config_cascade.json").read_text())
pm = {"root_cause":"bad_config_deploy","affected_services":["config-service","auth-service","redis","app-service","api-gateway"],
      "timeline_steps":4,"prevention_steps":["config_validation_ci","canary_deploy","staging_parity"]}
chk("hard perfect", grade_hard(1.0, pm, st(diagnosed=True,diagnosed_service="config-service"), s) == 1.0)
chk("hard zero", grade_hard(0.0, {}, st(), s) == 0.0)

chk("mttr fast", compute_mttr_bonus(1.0, 1, 15) > 0.9)
chk("mttr in range", all(0<=compute_mttr_bonus(b,s,15)<=1 for b in [0,0.5,1] for s in [1,7,15]))

# ── Scenario integrity ──
print("\n=== SCENARIOS ===")
STEPS = {"easy":10,"medium":15,"hard":20}
REQ = {"scenario_id","difficulty","services","dependency_graph","fault_service",
       "fault_type","fault_severity","playbook_effects","available_playbooks",
       "service_logs","red_herrings","max_steps"}
for fn in sorted(SCENARIOS_DIR.glob("*.json")):
    s = json.loads(fn.read_text())
    ok = (not (REQ - set(s.keys()))
          and s["fault_service"] in s["services"]
          and s["max_steps"] == STEPS[s["difficulty"]]
          and all(r["service"] != s["fault_service"] for r in s.get("red_herrings",[])))
    chk(fn.name, ok)

# ── Scenario generator ──
print("\n=== GENERATOR ===")
from server.scenario_generator import generate_scenario
for d in ("easy","medium","hard"):
    sc = generate_scenario(d, seed=42)
    chk(f"gen_{d}", sc["difficulty"]==d and sc["correct_playbook"] in sc["available_playbooks"])

# ── Environment integration ──
print("\n=== ENVIRONMENT ===")
try:
    from server.sre_environment import SREEnvironment
    from models import SREAction
    env = SREEnvironment()
    obs = env.reset(task_id="task_easy_1")
    chk("reset returns obs", obs is not None and not obs.done)
    chk("has alerts", len(obs.alerts) > 0)
    chk("has health metric", "system_avg_health" in obs.metrics)
    h0 = obs.metrics["system_avg_health"]
    chk("initial health < 1.0", h0 < 1.0, f"health={h0:.3f}")

    # Query + diagnose + remediate root cause
    obs = env.step(SREAction(action_type="query_logs", payload={"service": "redis"}))
    obs = env.step(SREAction(action_type="submit_diagnosis",
        payload={"suspected_service": "redis", "suspected_cause": "memory_leak"}))
    chk("correct diagnosis positive", obs.reward > 0)
    obs = env.step(SREAction(action_type="apply_remediation", payload={"playbook_id": "restart_redis"}))
    chk("root cause fix positive", obs.reward > 0)
    chk("fault_fixed=True", obs.metadata.get("fault_fixed") == True)

    # Let propagation recover + submit postmortem
    obs = env.step(SREAction(action_type="query_metrics", payload={"service": "api-gateway"}))
    h1 = obs.metrics.get("system_avg_health", 0)
    chk("health recovering", h1 > h0, f"before={h0:.3f} after={h1:.3f}")

    obs = env.step(SREAction(action_type="submit_postmortem", payload={
        "root_cause":"memory_leak","affected_services":["redis","app-service","api-gateway"],
        "timeline_steps":3,"prevention_steps":["redis_memory_alert"]}))
    chk("postmortem terminates", obs.done)
    chk("final reward > 0", obs.reward > 0, f"reward={obs.reward}")

    # Test wrong remediation
    env2 = SREEnvironment()
    env2.reset(task_id="task_easy_1")
    obs2 = env2.step(SREAction(action_type="apply_remediation", payload={"playbook_id": "restart_app"}))
    chk("wrong remediation negative", obs2.reward <= 0)

except ImportError:
    chk("env import", False, "openenv-core not installed")

passed = sum(results)
total = len(results)
print(f"\n{'='*44}")
print(f"  Results: {passed}/{total} passed")
if passed == total:
    print("  \033[92mALL CHECKS PASSED\033[0m")
else:
    print(f"  \033[91m{total-passed} FAILED\033[0m")
    sys.exit(1)
