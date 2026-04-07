import json
from pathlib import Path

d = Path("scenarios")
files = sorted(d.glob("*.json"))
errors = []

for f in files:
    s = json.loads(f.read_text())
    graph = s["dependency_graph"]
    all_svcs = set(graph.keys()) | {sv for deps in graph.values() for sv in deps}
    rcs = s["root_cause_service"]
    if rcs not in all_svcs:
        errors.append(f"{f.name}: root_cause_service={rcs!r} NOT in graph")
    cp = s["correct_playbook"]
    if cp not in s["available_playbooks"]:
        errors.append(f"{f.name}: correct_playbook={cp!r} NOT in available_playbooks")
    for rh in s.get("red_herrings", []):
        if rh["service"] == rcs:
            errors.append(f"{f.name}: root_cause_service appears as a red herring!")

if errors:
    for e in errors:
        print("FAIL:", e)
else:
    print(f"ALL {len(files)} SCENARIOS VALID")
    for f in files:
        print(f"  OK: {f.name}")
