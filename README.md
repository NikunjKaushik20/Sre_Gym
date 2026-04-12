---
title: SRE Gym
emoji: "🚨"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

<div align="center">
  <img src="https://img.shields.io/badge/OpenEnv-Compatible-blue.svg?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Docker-Ready-green?style=for-the-badge&logo=docker" />
  <img src="https://img.shields.io/badge/Gymnasium-Compatible-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PPO-Trained-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Grading-Deterministic-red?style=for-the-badge" />
</div>

<h1 align="center">🚨 SRE-Gym</h1>

<p align="center">
  <strong>A deterministic RL environment for production incident response — dependency graph traversal, cascading failure simulation, adversarial red herrings, and the only SRE benchmark with a <code>submit_postmortem</code> mechanic and zero LLM judges.</strong>
</p>

---

## Why This Exists

Every production outage shares the same structure: one broken service propagates degradation through a dependency graph, and the cascade makes every downstream service look like a root cause. An SRE's job is not to restart the loudest pod — it's to traverse the graph, reject noise, and write a postmortem that prevents recurrence.

No existing RL benchmark measures this. Most simulate isolated faults or use LLM judges that are non-reproducible, expensive, and gameable. SRE-Gym gives you:

- **The real SRE workflow**: Observe → Investigate → Traverse graph → Diagnose root (not symptom) → Remediate → Write postmortem
- **Deterministic graders**: Every score is reproducible in microseconds with no API keys
- **Learnable rewards**: A PPO agent improves measurably; the environment is provably trainable

---

## Multi-Model Leaderboard

| Agent | Easy | Medium | Hard | Overall |
|---|:---:|:---:|:---:|:---:|
| 🥇 Graph-aware BFS agent (`agent_graph.py`) | **0.878** | **0.864** | **0.886** | **0.876** |
| 🥈 PPO Agent (trained, 30K steps) | 0.643 | — | — | — |
| 🤖 gpt-4o-mini | 0.472 | 0.514 | 0.508 | 0.498 |
| 🔧 Deterministic keyword baseline | 0.38  | 0.21  | 0.09  | 0.23  |
| 🎲 Random agent | 0.134 | 0.08  | 0.04  | 0.08  |

> **Note on blanks:** The PPO agent is only trained and evaluated on Easy tasks. Medium/Hard tasks require generating a free-text unstructured postmortem, which is designed specifically to test LLMs, not standard discrete-action RL agents.
> The topological rule-based BFS agent achieves 0.88 overall, demonstrating that perfect performance requires graph-awareness rather than simple semantic matching (where LLMs excel but fail here).

---

## Key Differentiators

**Deterministic grading is the most important design choice here.**

Most agentic environments rely heavily on LLM judges. LLM evaluators are:
- Non-reproducible (same input → different score on re-run)
- Expensive ($0.01–$0.05 per evaluation)
- Prompt-sensitive (rephrasing changes the score)
- Gameable (verbosity inflation, hedging, keyword stuffing)

SRE-Gym graders run in **< 1ms**, produce **identical scores on identical inputs**, require **no API keys**, and use **fuzzy-match with a 0.92 SequenceMatcher threshold** — strict enough that keyword stuffing yields zero credit.

### Core Mechanics
- **Deterministic graders**: No LLM in the loop for evaluation.
- **Dependency graph traversal**: Agents must navigate microservice topologies, not just text.
- **`submit_postmortem` mechanic**: The ultimate test of incident comprehension.
- **Red herring injection**: Tests the agent's ability to isolate causal chains.
- **Procedural generation**: Infinite variations of failure scenarios.
- **Gymnasium wrapper + PPO**: Fully compatible with standard RL libraries.
- **MTTR-based scoring**: Emulates real-world SLA pressures.
- **Partial observability**: Agents only see what they explicitly query.

---

## Scoring Design Rationale

Every weight is deliberate, not arbitrary:

| Component | Easy | Medium | Hard | Why |
|---|:---:|:---:|:---:|---|
| Health recovery | 20% | 15% | 10% | Health is a **symptom**, not the goal. A 20% cap forces actual diagnosis rather than random healing |
| Correct diagnosis | 30% | 30% | 10% | Both service AND cause required. Half credit for service alone = 0 |
| Correct remediation | 25% | 15% | 0% | Gated on full correct diagnosis — lucky fix without diagnosis = 0 |
| Postmortem quality | 0% | 0% | 80% | On Hard, the postmortem **IS** the deliverable. Root cause, affected services, depth, prevention |
| MTTR efficiency | 25% | 40% | 0% | Medium tasks have SLA pressure. Fast-but-wrong still fails. Hard tasks have no MTTR because postmortem quality is the signal |
| Red herring penalty | 0 | −0.40 | −0.20 | Medium penalty (−0.40) exceeds remediation bonus (+0.15) — chasing noise is explicitly worse than doing nothing |

**On Hard tasks**, the scoring shift from diagnosis/remediation to postmortem quality mirrors how real SRE orgs work: a P0 incident requires a written postmortem that will be reviewed by the team. Writing a correct postmortem requires understanding the full blast radius — not just the root service.

---

## Baseline Benchmarks

| Scenario | Difficulty | gpt-4o-mini | Graph-Aware Agent |
|---|:---:|:---:|:---:|
| Redis OOM | Easy | 0.427 | 0.878 |
| Postgres Pool Exhaustion | Easy | 0.429 | 0.880 |
| Config Typo CrashLoop | Easy | 0.475 | 0.986 |
| Disk Full | Easy | 0.557 | 0.766 |
| Cascade Redis | Medium | 0.556 | 0.838 |
| Auth JWT Bug | Medium | 0.586 | 0.839 |
| Traffic Spike | Medium | 0.325 | 0.941 |
| TLS Cert Expiry (mTLS) | Medium | 0.588 | 0.839 |
| Config Cascade | Hard | 0.385 | 0.986 |
| Network Partition | Hard | 0.577 | 0.903 |
| Deploy Regression | Hard | 0.497 | 0.753 |
| DNS Misconfiguration | Hard | 0.573 | 0.903 |

> Scores reflect terminal-only grading (only the final `submit_postmortem` reward counts). A pure LLM scoring 0.472 on Easy is **not** doing well — navigating the topology with graph logic vastly outperforms plain text reasoning.

---

## Episode Flow

```
reset(task_id)
     │
     ▼
[OBSERVE]  Receive initial alerts + partial topology view
     │
     ▼
[INVESTIGATE]  query_logs / query_metrics
     │         (each query expands the visible dependency graph)
     ▼
[TRAVERSE]  Build directed graph → find topological root of unhealthy subgraph
     │      (root = unhealthy node with no unhealthy upstream dependency)
     ▼
[DIAGNOSE]  submit_diagnosis(service, cause)  ← exact canonical name required
     │
     ▼
[REMEDIATE] apply_remediation(playbook_id) ← root fix triggers cascade recovery
     │
     ▼
[POSTMORTEM] submit_postmortem(root_cause, affected_services,
     │                         timeline_steps, prevention_steps)
     ▼
[SCORE]  Deterministic grader → MTTR bonus → terminal reward ∈ (0.01, 0.99)
```

---

## Quickstart

### Docker (Recommended)
```bash
docker build -t sre-gym -f Dockerfile .
docker run -p 8000:8000 sre-gym
```

### Local Python
```bash
pip install -e ".[dev]"
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run Deterministic Baseline (No API Keys)
```bash
python baseline_deterministic.py
```

### Run Graph-Aware Agent (BFS Topology Traversal)
```bash
python agent_graph.py --verbose
python agent_graph.py --task task_hard_2
```

### Train PPO Agent
```bash
pip install stable-baselines3 matplotlib scipy tensorboard
python train_ppo.py                   # 100K steps, saves reward_curve.png
python train_ppo.py --steps 50000    # faster test run
```

### Run LLM Baseline
```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_api_key
python inference.py
```

---

## Observation & Action API

### Observation Space
| Field | Description |
|---|---|
| `alerts` | Active PagerDuty-style alerts `[id, service, severity, message]` |
| `logs` / `metrics` | Queried logs and floating-point metrics (evolve over time) |
| `dependency_graph` | Discovered portion of the microservice topology (partial observability) |
| `incident_state` | Boolean flags: `acknowledged`, `diagnosed`, `remediated`, `resolved` |
| `available_playbooks` | List of playbook IDs the agent can apply |

### Action Space
```json
{"action_type": "<type>", "payload": {<params>}}
```
- `query_logs` — Retrieve logs for a named service (+0.02 if fault service)
- `query_metrics` — Fetch live metrics for a service (+0.02 if fault service)
- `submit_diagnosis` — Identify root cause service AND failure mode (exact canonical name)
- `apply_remediation` — Execute a remediation playbook (only effective on root service)
- `escalate` — Escalate to an on-call team
- `submit_postmortem` — Provide timeline, root cause, affected services, prevention steps
- `close_incident` — End episode early (50% score penalty)

### Gymnasium Interface (for RL training)
```python
from gym_wrapper import SREGymEnv
from stable_baselines3 import PPO

env = SREGymEnv("task_easy_1", seed=42)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

---

## Architecture

```
sre_gym/
├── server/
│   ├── app.py                 — FastAPI + dashboard serving
│   ├── sre_environment.py     — Core step/reset/state machine
│   ├── graders.py             — Deterministic scoring (0.92-threshold fuzzy match)
│   ├── scenario_generator.py  — Procedural scenario generation
│   └── static/
│       └── dashboard.html     — Live incident command center
├── scenarios/                 — 12 curated JSON scenarios
├── models.py                  — Pydantic Action/Observation/State
├── gym_wrapper.py             — Gymnasium env (stable-baselines3 compatible)
├── train_ppo.py               — PPO training + reward curve generation
├── agent_graph.py             — Graph-aware BFS agent (topology traversal)
├── inference.py               — LLM baseline script
├── baseline_deterministic.py  — Rule-based baseline (no API keys)
├── client.py                  — Typed async client
├── openenv.yaml               — OpenEnv specification
├── Dockerfile                 — Multi-stage Docker build
└── pyproject.toml             — Hatchling build config
```

### Scoring Architecture

```
submit_postmortem()
    │
    ├── grade_easy()   → health(20%) + diagnosis(30%) + remediation(25%) + MTTR(25%)
    ├── grade_medium() → health(15%) + diagnosis(30%) + remediation(15%) + MTTR(40%)
    │                    red_herring_penalty(−40%)
    └── grade_hard()   → health(10%) + diagnosis(10%) + postmortem(80%)
                         [root_cause(25%) + affected(15%) + timeline(15%) + prevention(25%)]
         │
         └── compute_mttr_bonus() → score = base×0.80 + efficiency×0.20
                                    (speed cannot compensate for wrong answer)
```

---

## Running Tests
```bash
pip install -e ".[dev]"
pytest tests/ -v
python agent_graph.py        # integration smoke test
python gym_wrapper.py        # gymnasium spec check
```

## License
MIT
