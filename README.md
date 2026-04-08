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

<h1 align="center">🚨 SRE-Gym</h1>

<p align="center">
  <strong>A deterministic RL environment for production incident response — featuring cascading distributed system failures, dependency graph reasoning, dense reward shaping, and MTTR-based scoring.</strong>
</p>

---

## Overview

SRE-Gym simulates real-world production outages where AI agents must diagnose cascading failures across microservice architectures. In production engineering, an SRE (Site Reliability Engineer) is the person paged at 3 a.m. to find which service in a chain of 20 is actually on fire — this benchmark measures whether an AI can do that job. Agents interact through structured actions (querying logs, querying metrics, submitting diagnoses, applying remediations) and are scored by fully deterministic graders — no LLM judges.

**Key differentiators:**
- **Dependency Graph Reasoning** — Agents traverse microservice topologies to distinguish root causes from downstream victims
- **Red Herring Injection** — Decoy alerts and metrics penalize models that chase noise
- **Dynamic Metric Evolution** — Metrics worsen each step without remediation, creating real MTTR pressure
- **Partial Observability** — Dependency graph revealed incrementally as agents query services
- **Strict Grading** — Exact canonical cause names required; substring guessing yields zero credit
- **Procedural Scenario Generation** — Infinite task variation from composable building blocks
- **MTTR Scoring** — Faster resolution yields higher scores, mirroring real SRE KPIs

---

## Baseline Benchmarks

| Scenario | Difficulty | `gpt-4o-mini` |
|---|:---:|:---:|
| Redis OOM | Easy | 0.427 |
| Postgres Pool Exhaustion | Easy | 0.429 |
| Config Typo CrashLoop | Easy | 0.475 |
| Disk Full | Easy | 0.557 |
| Cascade Redis | Medium | 0.556 |
| Auth JWT Bug | Medium | 0.586 |
| Traffic Spike | Medium | 0.325 |
| TLS Cert Expiry (mTLS) | Medium | 0.588 |
| Config Cascade | Hard | 0.385 |
| Network Partition | Hard | 0.577 |
| Deploy Regression | Hard | 0.497 |
| DNS Misconfiguration | Hard | 0.573 |
| **Easy avg** | | **0.472** |
| **Medium avg** | | **0.514** |
| **Hard avg** | | **0.508** |
| **Overall avg** | | **0.498** |

> Scores reflect terminal-only grading (only the final `submit_postmortem` reward counts; intermediate step signals are excluded from the benchmark score).

> The deterministic baseline (`python baseline_deterministic.py`) is a rule-based keyword-matching agent that reads alerts and logs without any LLM — it establishes a lower-bound floor that any serious model should exceed.

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
| `dependency_graph` | Discovered portion of the microservice topology |
| `incident_state` | Boolean flags: `acknowledged`, `diagnosed`, `remediated`, `resolved` |
| `available_playbooks` | List of playbook IDs the agent can apply |

### Action Space
```json
{"action_type": "<type>", "payload": {<params>}}
```
- `query_logs` — Retrieve logs for a named service
- `query_metrics` — Fetch live metrics for a service
- `submit_diagnosis` — Identify root cause service and failure mode
- `apply_remediation` — Execute a remediation playbook
- `escalate` — Escalate to an on-call team
- `submit_postmortem` — Provide timeline, root cause, and prevention plan

---

## Scoring System

| Component | Easy | Medium | Hard |
|---|:---:|:---:|:---:|
| Health recovery (capped) | 20% | 15% | 10% |
| Correct diagnosis (service + cause) | 30% | 30% | 10% |
| Correct remediation | 25% | 15% | — |
| Postmortem: root cause | — | — | 25% |
| Postmortem: affected services (≥80%) | — | — | 15% |
| Postmortem: timeline depth (min+3) | — | — | 15% |
| Postmortem: prevention steps (≥75%) | — | — | 25% |
| Red herring penalty | — | −0.40 | −0.20 |
| MTTR efficiency bonus | 25% | 40% | n/a |
| **Max score** | **1.0** | **1.0** | **1.0** |

Graders are **strict**: exact canonical cause names required (e.g. `cert_expiry`, not `certificate error`); no substring or partial-phrase credit.

---

## Architecture

```
sre_gym/
├── server/
│   ├── app.py                — FastAPI + dashboard serving
│   ├── sre_environment.py    — Core step/reset/state machine
│   ├── graders.py            — Fuzzy deterministic scoring
│   ├── scenario_generator.py — Procedural scenario generation
│   └── static/
│       └── dashboard.html    — Incident command center UI
├── scenarios/                — 12 curated JSON scenarios
├── models.py                 — Pydantic Action/Observation/State
├── client.py                 — Typed async client
├── inference.py              — LLM baseline script
├── baseline_deterministic.py — Rule-based baseline (no API keys)
├── openenv.yaml              — OpenEnv specification
├── Dockerfile                — Multi-stage Docker build
└── pyproject.toml            — Hatchling build config
```

---

## Running Tests
```bash
pip install -e ".[dev]"
pytest tests/ -v
python smoke_test.py
```

## License
MIT
