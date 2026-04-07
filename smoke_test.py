"""Quick smoke-test of graders and scenario integrity without openenv-core."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

root = Path(".")
sys.path.insert(0, str(root / "server"))

from graders import grade_easy, grade_medium, grade_hard, compute_mttr_bonus

SCENARIOS_DIR = root / "scenarios"
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append(condition)
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

def load(fn):
    return json.loads((SCENARIOS_DIR / fn).read_text())

def state(**kw):
    defaults = dict(
        diagnosed=False, diagnosed_service=None, diagnosed_cause=None,
        remediated=False, remediation_applied=None,
        escalated=False, escalation_team=None, acknowledged=False,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)

print("\n=== GRADER TESTS ===")

# grade_easy
s = load("easy_redis_oom.json")
check("grade_easy perfect",
      grade_easy(state(diagnosed=True, diagnosed_service="redis", diagnosed_cause="memory_leak",
                       remediated=True, remediation_applied="restart_redis"), s) == 1.0)
check("grade_easy zero",
      grade_easy(state(), s) == 0.0)
check("grade_easy wrong service < 0.5",
      grade_easy(state(diagnosed=True, diagnosed_service="postgres", diagnosed_cause="memory_leak",
                       remediated=True, remediation_applied="restart_redis"), s) < 0.8)
check("grade_easy wrong playbook < 1.0",
      grade_easy(state(diagnosed=True, diagnosed_service="redis", diagnosed_cause="memory_leak",
                       remediated=True, remediation_applied="scale_postgres_pool"), s) < 1.0)

# grade_medium
s = load("medium_cascade_redis.json")
check("grade_medium perfect",
      grade_medium(state(diagnosed_service="redis", diagnosed_cause="memory_leak",
                         remediation_applied="restart_redis", escalated=False), s) == 1.0)
check("grade_medium red herring penalty",
      grade_medium(state(diagnosed_service="postgres", escalated=False), s) < 0.3)

# grade_hard
s = load("hard_config_cascade.json")
perfect_payload = {
    "root_cause": "bad_config_deploy",
    "affected_services": ["auth-service", "redis", "api-gateway"],
    "timeline_steps": 4,
    "prevention_steps": ["test config in staging", "canary deploy"],
}
check("grade_hard perfect", grade_hard(perfect_payload, state(), s) == 1.0)
wrong_payload = {**perfect_payload, "root_cause": "wrong"}
check("grade_hard wrong root_cause < 0.7", grade_hard(wrong_payload, state(), s) < 0.7)

# MTTR
check("mttr fast 1-step > 0.9",   compute_mttr_bonus(1.0, 1, 15) > 0.9)
check("mttr full-steps ≈ 0.75",   abs(compute_mttr_bonus(1.0, 15, 15) - 0.75) < 0.01)
check("mttr zero base → 0.0+",    compute_mttr_bonus(0.0, 5, 15) >= 0.0)
check("mttr always in [0,1]",     all(
    0.0 <= compute_mttr_bonus(b, s, 15) <= 1.0
    for b in [0.0, 0.5, 1.0] for s in [1, 7, 15]
))

print("\n=== SCENARIO INTEGRITY ===")
for fn in sorted(SCENARIOS_DIR.glob("*.json")):
    s = json.loads(fn.read_text())
    graph = s["dependency_graph"]
    all_svcs = set(graph.keys()) | {sv for deps in graph.values() for sv in deps}
    ok = (s["root_cause_service"] in all_svcs
          and s["correct_playbook"] in s["available_playbooks"]
          and all(rh["service"] != s["root_cause_service"] for rh in s.get("red_herrings", [])))
    check(fn.name, ok)

passed = sum(results)
total  = len(results)
print(f"\n{'='*44}")
print(f"  Results: {passed}/{total} passed")
if passed == total:
    print("  \033[92mALL CHECKS PASSED ✓\033[0m")
else:
    sys.exit(1)
