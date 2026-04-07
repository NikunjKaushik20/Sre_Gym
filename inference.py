"""Baseline inference script for SRE-Gym.

Uses OpenAI-compatible client to solve SRE incidents.
Emits structured [START]/[STEP]/[END] logs for automated evaluation.
"""
import os
import json
import requests

from openai import OpenAI

# ── Configuration from environment ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

SYSTEM_PROMPT = """You are an expert SRE diagnosing a production incident.
Think step by step:
1. Read the dependency graph — downstream victims are NOT root causes.
2. Find the service with the earliest anomaly timestamp.
3. Check if any alert is a known red herring (slow queries during off-peak = likely unrelated).
4. Query logs for the most suspicious service first.
5. Submit diagnosis only when confident.
6. Apply the minimal correct remediation from available playbooks.
7. For hard tasks, submit a complete postmortem.

ALWAYS respond with valid JSON matching this schema:
{"action_type": "<type>", "payload": {<params>}}

Valid action_types: query_logs, query_metrics, submit_diagnosis, apply_remediation, escalate, submit_postmortem
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
    parts = [
        f"=== INCIDENT STATUS (step {obs.get('incident_state', {}).get('step', '?')}"
        f"/{obs.get('incident_state', {}).get('max_steps', '?')}) ===",
        f"Message: {obs.get('message', '')}",
        f"\nALERTS:",
    ]
    for a in obs.get("alerts", []):
        parts.append(f"  [{a['severity'].upper()}] {a['service']}: {a['message']} (t={a['timestamp']})")
    parts.append(f"\nLOGS (last {len(obs.get('logs', []))} entries):")
    for l in obs.get("logs", []):
        parts.append(f"  [t={l['t']}] {l['service']}: {l['msg']}")
    parts.append(f"\nMETRICS: {json.dumps(obs.get('metrics', {}), indent=2)}")
    parts.append(f"\nDEPENDENCY GRAPH: {json.dumps(obs.get('dependency_graph', {}), indent=2)}")
    parts.append(f"\nAVAILABLE PLAYBOOKS: {obs.get('available_playbooks', [])}")
    parts.append(f"\nINCIDENT STATE: {json.dumps(obs.get('incident_state', {}), indent=2)}")
    return "\n".join(parts)


def parse_action(response_text):
    """Extract JSON action from LLM response."""
    text = response_text.strip()
    # Try to find JSON in the response
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise


def run_task(client, task_info):
    """Run a single task and return the final score."""
    task_id = task_info["task_id"]
    task_name = task_info["name"]
    print(f"[START] task_id={task_id} task_name={task_name}")

    total_reward = 0.0
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    with SREGymEnv(base_url=ENV_URL).sync() as env:
        # Reset environment
        step_result = env.reset(task_id=task_id)
        obs = step_result.observation.model_dump()

        for step_num in range(1, 16):
            # Format observation for LLM
            obs_text = format_observation(obs)
            messages.append({"role": "user", "content": obs_text})

            # Call LLM
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=500,
                )
                llm_output = response.choices[0].message.content
                messages.append({"role": "assistant", "content": llm_output})
            except Exception as e:
                print(f"[STEP] step={step_num} error=LLM call failed: {e}")
                break

            # Parse action
            try:
                action_dict = parse_action(llm_output)
            except Exception as e:
                print(f"[STEP] step={step_num} error=Failed to parse action: {e}")
                action_dict = {"action_type": "query_logs", "payload": {"service": "redis"}}

            try:
                action_model = SREAction(**action_dict)
            except Exception as e:
                print(f"[STEP] step={step_num} error=Invalid action schema: {e}")
                action_model = SREAction(action_type="query_logs", payload={"service": "redis"})

            # Step environment
            step_result = env.step(action_model)
            obs = step_result.observation.model_dump()
            reward = step_result.reward
            done = step_result.done
            total_reward += reward

            print(
                f"[STEP] step={step_num} "
                f"action={action_model.action_type} "
                f"reward={reward} "
                f"cumulative_reward={round(total_reward, 3)} "
                f"done={done}"
            )

            if done:
                break

    # Strictly clamp final episode bounds to [0.0, 1.0] to prevent reward looping arithmetic
    total_reward = max(0.0, min(total_reward, 1.0))

    print(f"[END] task_id={task_id} total_reward={round(total_reward, 3)}")
    return total_reward


def main():
    """Run baseline inference across all tasks."""
    client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)
    results = {}

    for task_info in TASKS:
        score = run_task(client, task_info)
        results[task_info["task_id"]] = score

    print("\n=== BASELINE RESULTS ===")
    for tid, score in results.items():
        print(f"  {tid}: {score}")
    print(f"  Average: {sum(results.values()) / len(results):.3f}")


if __name__ == "__main__":
    main()
