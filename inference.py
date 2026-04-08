"""Baseline inference script for SRE-Gym.

Uses OpenAI-compatible client to solve SRE incidents.
Emits structured [START]/[STEP]/[END] logs for automated evaluation.

Scoring discipline:
  - ONLY the terminal reward (from submit_postmortem or timeout) counts.
  - Step-level intermediate rewards (+0.15 diagnosis, +0.25 remediation) are
    deliberately discarded; they exist to shape agent behaviour, not inflate
    the final score.
  - This mirrors how human evaluators score postmortems: the journey matters
    for learning, but only the final written artefact is graded.
"""
import os
import json
import requests

from openai import OpenAI

# ── Configuration from environment ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer responding to a production incident.

## MANDATORY REASONING PROTOCOL (follow this order every episode):

### Phase 1 — Reconnaissance (do NOT diagnose yet)
1. Read every alert. Note services and timestamps.
2. Read the dependency graph. DOWNSTREAM victims are NOT root causes.
   - A → B → C: if C is the earliest alert, C caused A and B.
3. Query logs for the service with the EARLIEST critical/fatal alert timestamp.
4. Query metrics for that service.
5. If any service shows an alert marked "(red herring)" or "unrelated" in logs, skip it.

### Phase 2 — Diagnosis (only after querying ≥2 services)
6. submit_diagnosis with EXACT cause type from this list:
   memory_leak | pool_exhaustion | config_typo | disk_full | jwt_bug |
   traffic_spike | cert_expiry | bad_config_deploy | deploy_regression |
   dns_misconfiguration | network_partition
   DO NOT submit generic causes like "config", "error", "high memory".

### Phase 3 — Remediation
7. apply_remediation with the playbook targeting the ROOT service.
   Do NOT apply playbooks to downstream victims.

### Phase 4 — Postmortem (required — produces the graded score)
8. submit_postmortem with ALL of these fields:
   {
     "root_cause": "<exact cause type from Phase 2>",
     "affected_services": ["<service1>", "<service2>", ...],  // list ALL affected
     "timeline_steps": [
       "t=0: <earliest alert>",
       "t=1: <log evidence>",
       "t=2: <propagation observed>",
       "t=3: <diagnosis submitted>",
       "t=4: <remediation applied>"
     ],  // list ≥5 specific timestamped steps
     "prevention_steps": ["<specific action 1>", "<specific action 2>", "<specific action 3>"]
     // must name real SRE practices (e.g. "add Redis maxmemory alert at 80%")
   }

## STRICT RULES:
- ALWAYS respond with ONLY valid JSON. No prose, no markdown, no explanation.
- Format: {"action_type": "<type>", "payload": {<params>}}
- Valid action_types: query_logs, query_metrics, submit_diagnosis,
  apply_remediation, escalate, submit_postmortem
- Prevention steps must be SPECIFIC (not "add monitoring" — say "add OOM alert on Redis").
- timeline_steps must be a LIST of strings, each describing one event.

## WHAT WILL HURT YOUR SCORE:
- Diagnosing a red herring service: -0.40 penalty
- Generic root cause labels (e.g. "config_error" instead of "bad_config_deploy"): 0 credit
- Affected services < 80% overlap with actual: 0 credit
- Timeline with fewer than (min_required + 3) steps: 0 credit
- Prevention steps < 75% overlap with valid set: 0 credit
"""

TASKS = [
    # Easy (4)
    {"task_id": "task_easy_1",   "name": "Easy: Redis OOM"},
    {"task_id": "task_easy_2",   "name": "Easy: Postgres Pool Exhaustion"},
    {"task_id": "task_easy_3",   "name": "Easy: Config Typo CrashLoop"},
    {"task_id": "task_easy_4",   "name": "Easy: Disk Full"},
    # Medium (4)
    {"task_id": "task_medium_1", "name": "Medium: Cascade Redis"},
    {"task_id": "task_medium_2", "name": "Medium: Auth JWT Bug"},
    {"task_id": "task_medium_3", "name": "Medium: Traffic Spike"},
    {"task_id": "task_medium_4", "name": "Medium: TLS Cert Expiry"},
    # Hard (4)
    {"task_id": "task_hard_1",   "name": "Hard: Config Cascade"},
    {"task_id": "task_hard_2",   "name": "Hard: Network Partition"},
    {"task_id": "task_hard_3",   "name": "Hard: Deploy Regression"},
    {"task_id": "task_hard_4",   "name": "Hard: DNS Misconfiguration"},
]


# Ensure 'sre_gym' package is discoverable from parent directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sre_gym.client import SREGymEnv
from sre_gym.models import SREAction

def format_observation(obs):
    """Format observation as readable string for the LLM."""
    inc = obs.get('incident_state', {})
    parts = [
        f"=== INCIDENT STATUS (step {inc.get('step', '?')}/{inc.get('max_steps', '?')}) ===",
        f"Message: {obs.get('message', '')}",
        "",
        "ALERTS (sorted by timestamp):",
    ]
    alerts = sorted(obs.get("alerts", []), key=lambda a: a.get("timestamp", 0))
    for a in alerts:
        parts.append(f"  [{a['severity'].upper()}] t={a['timestamp']} {a['service']}: {a['message']}")

    parts.append(f"\nLOGS (last {len(obs.get('logs', []))} entries):")
    for l in obs.get("logs", []):
        parts.append(f"  [t={l['t']}][{l['level'].upper()}] {l['service']}: {l['msg']}")

    parts.append(f"\nLIVE METRICS: {json.dumps(obs.get('metrics', {}), indent=2)}")
    parts.append(f"\nDEPENDENCY GRAPH (partial — query more services to reveal more):")
    for svc, deps in obs.get('dependency_graph', {}).items():
        parts.append(f"  {svc} → {deps}")

    parts.append(f"\nAVAILABLE PLAYBOOKS: {obs.get('available_playbooks', [])}")
    parts.append(f"\nINCIDENT STATE: diagnosed={inc.get('diagnosed')}, "
                 f"remediated={inc.get('remediated')}, "
                 f"resolved={inc.get('resolved')}")
    return "\n".join(parts)


def parse_action(response_text):
    """Extract JSON action from LLM response. Strict: reject non-JSON prose."""
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last-resort: find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise


def run_task(client, task_info):
    """Run a single task and return the TERMINAL reward only.

    Critical change: intermediate step rewards are NOT summed into the final
    score. Only the reward from the terminal action (submit_postmortem or
    timeout) is returned. This prevents inflation from +0.15 diagnosis and
    +0.25 remediation step signals.
    """
    task_id = task_info["task_id"]
    task_name = task_info["name"]
    print(f"[START] task_id={task_id} task_name={task_name}")

    terminal_reward = 0.0
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    with SREGymEnv(base_url=ENV_URL).sync() as env:
        step_result = env.reset(task_id=task_id)
        obs = step_result.observation.model_dump()

        for step_num in range(1, 22):  # hard cap at 21 to cover hard (max_steps=20)
            obs_text = format_observation(obs)
            messages.append({"role": "user", "content": obs_text})

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,    # deterministic — no lucky rolls
                    max_tokens=800,     # enough for full postmortem JSON
                )
                llm_output = response.choices[0].message.content
                messages.append({"role": "assistant", "content": llm_output})
            except Exception as e:
                print(f"[STEP] step={step_num} error=LLM call failed: {e}")
                break

            try:
                action_dict = parse_action(llm_output)
            except Exception as e:
                print(f"[STEP] step={step_num} error=Failed to parse JSON: {e}")
                # Penalty: force a safe no-op rather than a lucky default
                action_dict = {"action_type": "query_logs", "payload": {"service": "api-gateway"}}

            try:
                action_model = SREAction(**action_dict)
            except Exception as e:
                print(f"[STEP] step={step_num} error=Invalid action schema: {e}")
                action_model = SREAction(action_type="query_logs", payload={"service": "api-gateway"})

            step_result = env.step(action_model)
            obs = step_result.observation.model_dump()
            step_reward = step_result.reward
            done = step_result.done

            print(
                f"[STEP] step={step_num} "
                f"action={action_model.action_type} "
                f"step_reward={step_reward:.3f} "
                f"done={done}"
            )

            if done:
                # Only the TERMINAL reward counts toward benchmark score
                terminal_reward = max(0.0, min(float(step_reward), 1.0))
                break

    print(f"[END] task_id={task_id} terminal_reward={round(terminal_reward, 3)}")
    return terminal_reward


def main():
    """Run baseline inference across all tasks."""
    client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)
    results = {}

    for task_info in TASKS:
        score = run_task(client, task_info)
        results[task_info["task_id"]] = score

    print("\n=== BASELINE RESULTS ===")
    easy_scores   = [v for k, v in results.items() if "easy"   in k]
    medium_scores = [v for k, v in results.items() if "medium" in k]
    hard_scores   = [v for k, v in results.items() if "hard"   in k]

    for tid, score in results.items():
        print(f"  {tid}: {score:.3f}")

    all_scores = list(results.values())
    print(f"\n  Easy   avg: {sum(easy_scores)/len(easy_scores):.3f}")
    print(f"  Medium avg: {sum(medium_scores)/len(medium_scores):.3f}")
    print(f"  Hard   avg: {sum(hard_scores)/len(hard_scores):.3f}")
    print(f"  Overall avg: {sum(all_scores)/len(all_scores):.3f}")


if __name__ == "__main__":
    main()
