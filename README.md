---
title: SRE Gym
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
---

<div align="center">
  <img src="https://img.shields.io/badge/OpenEnv-Compatible-blue.svg?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Docker-Ready-green?style=for-the-badge&logo=docker" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi" />
</div>

<h1 align="center">🚨 SRE-Gym: The Ultimate Incident Response RL Environment</h1>

<p align="center">
  <strong>"SRE-Gym is the first fully deterministic RL environment for production incident response — simulating cascading distributed system failures, dependency graphs, dense reward shaping, and MTTR scoring. No LLM judges. No toy problems. Just real SRE work."</strong>
</p>

---

## 🏆 Why SRE-Gym Wins (The Hackathon Pitch)

Traditional AI agents are tested on simple Q&A tasks or web browsing. But **production outages are chaotic, dynamic, and full of red herrings.** Creating autonomous Site Reliability Engineering (SRE) agents requires a crucible that can actually grade their reasoning. 

SRE-Gym bridges this gap by offering:
- **🧠 First-Class Dependency Graphs:** Agents can't just guess; they must traverse microservice topologies backward from the 503 error down to the true root cause.
- **🎣 Decoy "Red Herrings":** Noise injected into metrics explicitly punishes models that lack impulse control. Weak models fall for the bait; strong models trace the root cause.
- **⏱️ MTTR-Based Continuous Scoring:** Faster resolution yields higher efficiency bonuses, directly mirroring real-world SRE Key Performance Indicators (KPIs).
- **🔒 Highly Deterministic Gradings:** Absolutely zero "LLM Judges." The environment evaluates the schema, the playbook applied, and the postmortem using bounded math `[0.0, 1.0]`.

---

## 📊 State-of-the-Art Baseline Benchmarks

We natively integrated and ran API inference against SRE-Gym to establish standard baselines. Notice how smaller models get trapped by the environment's strict MTTR penalties and "Red Herring" decoy injections, proving SRE-Gym is a powerful and discerning evaluation metric!

| Scenario ID | Task Name | `gpt-4o-mini` 
|---|---|:---:|
| `task_easy_1` | Easy: Redis OOM | **1.0** 
| `task_easy_2` | Easy: Postgres Pool Exhaustion | **0.477** 
| `task_easy_3` | Easy: Config Typo CrashLoop | **0.933** 
| `task_easy_4` | Easy: Disk Full | **1.000** 
| `task_medium_1` | Medium: Cascade Redis | **0.505** 
| `task_medium_2` | Medium: Auth JWT Bug | **0.896** 
| `task_medium_3` | Medium: Traffic Spike | **1.000** 
| `task_medium_4` | Medium: TLS Cert Expiry | **0.996** 
| `task_hard_1` | Hard: Config Cascade | **0.556** 
| `task_hard_2` | Hard: Network Partition | **0.809999** 
| `task_hard_3` | Hard: Deploy Regression | **0.683** 
| `task_hard_4` | Hard: DNS Misconfiguration | **1.0** 
| **🏆 Average Score** | **All 12 Scenarios** | **0.821** 

---

## 🚀 Quickstart: Up and Running in 60 Seconds

SRE-Gym is natively architected for wide deployment, packaging its own custom Hatchling `pyproject` into a highly-cached multi-stage Docker build.

### 🐳 Deploy via Docker (Recommended)
```bash
# Build the optimized container
docker build -t sre-gym -f Dockerfile .

# Expose the API to the world
docker run -p 8000:8000 sre-gym
```

### 🐍 Local Python Installation
```bash
pip install -e ".[dev]"
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🛠️ The Observation & Action API

SRE-Gym exposes a gorgeous JSON schema for AI Agents to interact with via REST or WebSockets.

### What the Agent SEES (Observation Space)
| Field | Description |
|---|---|
| `alerts` | Active PagerDuty-style alerts `[id, service, severity, message]` |
| `logs` / `metrics` | Real-time queried logs and floating-point CPU/Mem readouts |
| `dependency_graph` | The full topology mapping of the entire microservice ecosystem |
| `incident_state` | Boolean flags: `acknowledged`, `diagnosed`, `remediated`, `resolved` |

### What the Agent DOES (Action Space)
Agents submit cleanly formatted JSON commands `{"action_type": "...", "payload": {...}}`:
- `query_logs` (Retrieve history for a named service)
- `query_metrics` (Fetch live stats for a node)
- `submit_diagnosis` (Identify the exact fault and the broken service)
- `apply_remediation` (Trigger shell playbooks like "restart_redis")
- `escalate` (Wake up an L3 on-call engineer)
- `submit_postmortem` (Provide a timeline and prevention checklist)

---

## 💻 Zero-to-Hero: Demo Run

Connecting an agent to the environment only takes 5 lines of code:

```python
import asyncio
from sre_gym import SREGymEnv

async def run():
    async with SREGymEnv(base_url="http://localhost:8000") as env:
        # Load a high-level incident
        obs = await env.reset(task_id="task_medium_1")
        
        # Traverse the dependency graph
        obs = await env.query_logs("redis")
        
        # Submit the fix to the deterministic grader
        obs = await env.apply_remediation("restart_redis")
        
        # Finalize and claim your reward!
        obs = await env.submit_postmortem(
            root_cause="memory_leak",
            affected_services=["redis", "api-gateway"],
            timeline_steps=3,
            prevention_steps=["redis_memory_alert"]
        )
        print(f"Final MTTR Score: {obs.reward} / 1.0")

asyncio.run(run())
```

---

## 🏗️ Architecture Under the Hood

```
sre_gym/
├── client.py            — Stateful WebSocket API Client Interface
├── inference.py         — End-to-end OpenAI/Groq compatible Benchmark Script 
├── models.py            — Strict Pydantic Action/Observation schemas
├── openenv.yaml         — 12 highly-curated production scenarios
├── server/
│   ├── sre_environment.py — The state machine, tracing metrics, logs & step logic
│   ├── graders.py       — Complex deterministic scoring (clamping, MTTR math, red herrings)
│   └── app.py           — FastAPI openenv lifecycle server
└── pyproject.toml       — Hatchling configuration for rapid Docker caching
```

## 💖 Contributing
Built to push the frontier of what autonomous systems can achieve. SRE-Gym is fully open-source. Fork it, build new chaotic scenarios, or deploy it into Hugging Face Spaces instantly.
