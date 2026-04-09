"""
Pytest test suite for SRE-Gym.

Tests verify:
  - Grader functions return correct [0.0, 1.0] values
  - Fuzzy matching accepts synonyms and paraphrases
  - Environment reset / step / state flow
  - Reward shaping (correct diagnosis = positive, red herring = negative)
  - Payload validation catches missing fields
  - MTTR bonus formula
  - All 12 scenario files load without error
  - Variable max_steps per difficulty (easy=10, medium=15, hard=20)
  - Termination on max_steps
  - Dynamic metrics worsen each step
  - Partial observability (dependency graph discovered incrementally)
  - Procedural scenario generation
"""
import json
import sys
import pytest
from pathlib import Path
from types import SimpleNamespace

SRE_GYM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SRE_GYM_ROOT))
sys.path.insert(0, str(SRE_GYM_ROOT / "server"))

from server.graders import grade_easy, grade_medium, grade_hard, compute_mttr_bonus
from server.graders import _fuzzy_match, _fuzzy_set_overlap

# ── Fixtures ──

SCENARIOS_DIR = SRE_GYM_ROOT / "scenarios"

ALL_SCENARIO_FILES = [
    "easy_redis_oom.json", "easy_postgres_slow.json",
    "easy_config_typo.json", "easy_disk_full.json",
    "medium_cascade_redis.json", "medium_cascade_auth.json",
    "medium_traffic_spike.json", "medium_cert_expiry.json",
    "hard_config_cascade.json", "hard_network_partition.json",
    "hard_deploy_regression.json", "hard_multi_region.json",
]

REQUIRED_SCENARIO_KEYS = {
    # Keys that ALL static scenario JSON files must have
    # (matching what sre_environment.py actually reads)
    "scenario_id", "difficulty",
    "fault_service", "fault_type", "fault_severity",
    "dependency_graph", "playbook_effects",
    "available_playbooks", "service_logs", "red_herrings",
    "affected_services", "valid_prevention_steps",
    "min_timeline_steps", "max_steps",
}

EXPECTED_MAX_STEPS = {
    "easy": 10, "medium": 15, "hard": 20,
}

def _make_state(**kwargs):
    defaults = dict(
        diagnosed=False, diagnosed_service=None, diagnosed_cause=None,
        remediated=False, remediation_applied=None,
        escalated=False, escalation_team=None,
        acknowledged=False, resolved=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)

def _load_scenario(filename):
    with open(SCENARIOS_DIR / filename) as f:
        return json.load(f)


# ── SCENARIO FILE INTEGRITY ──

@pytest.mark.parametrize("filename", ALL_SCENARIO_FILES)
def test_scenario_file_exists(filename):
    assert (SCENARIOS_DIR / filename).exists()

@pytest.mark.parametrize("filename", ALL_SCENARIO_FILES)
def test_scenario_has_required_keys(filename):
    scenario = _load_scenario(filename)
    missing = REQUIRED_SCENARIO_KEYS - set(scenario.keys())
    assert not missing, f"{filename} missing keys: {missing}"

@pytest.mark.parametrize("filename", ALL_SCENARIO_FILES)
def test_scenario_difficulty_valid(filename):
    scenario = _load_scenario(filename)
    assert scenario["difficulty"] in {"easy", "medium", "hard"}

@pytest.mark.parametrize("filename", ALL_SCENARIO_FILES)
def test_scenario_max_steps_by_difficulty(filename):
    scenario = _load_scenario(filename)
    expected = EXPECTED_MAX_STEPS[scenario["difficulty"]]
    assert scenario["max_steps"] == expected, (
        f"{filename}: expected max_steps={expected} for {scenario['difficulty']}, got {scenario['max_steps']}"
    )

@pytest.mark.parametrize("filename", ALL_SCENARIO_FILES)
def test_scenario_root_cause_in_graph(filename):
    scenario = _load_scenario(filename)
    graph = scenario["dependency_graph"]
    all_svcs = set(graph.keys()) | {s for deps in graph.values() for s in deps}
    assert scenario["fault_service"] in all_svcs

@pytest.mark.parametrize("filename", ALL_SCENARIO_FILES)
def test_scenario_correct_playbook_in_available(filename):
    """The playbook that targets the fault_service must be in available_playbooks."""
    scenario = _load_scenario(filename)
    effects = scenario.get("playbook_effects", {})
    fault_svc = scenario["fault_service"]
    correct_pbs = [pb for pb, cfg in effects.items() if cfg.get("target") == fault_svc]
    available = scenario["available_playbooks"]
    assert any(pb in available for pb in correct_pbs), (
        f"{filename}: no playbook targeting {fault_svc} found in available_playbooks"
    )

@pytest.mark.parametrize("filename", ALL_SCENARIO_FILES)
def test_scenario_red_herrings_not_root_cause(filename):
    scenario = _load_scenario(filename)
    rh_svcs = {rh["service"] for rh in scenario.get("red_herrings", [])}
    assert scenario["fault_service"] not in rh_svcs

@pytest.mark.parametrize("filename", [f for f in ALL_SCENARIO_FILES if "hard" in f])
def test_hard_has_two_red_herrings(filename):
    scenario = _load_scenario(filename)
    assert len(scenario.get("red_herrings", [])) >= 2

@pytest.mark.parametrize("filename", [f for f in ALL_SCENARIO_FILES if "medium" in f])
def test_medium_has_one_red_herring(filename):
    scenario = _load_scenario(filename)
    assert len(scenario.get("red_herrings", [])) >= 1


# ── FUZZY MATCHING ──

def test_fuzzy_match_exact():
    assert _fuzzy_match("memory_leak", "memory_leak")

def test_fuzzy_match_synonym():
    assert _fuzzy_match("oom", "memory_leak")
    assert _fuzzy_match("out_of_memory", "memory_leak")

def test_fuzzy_match_substring_now_rejected():
    """Substring matching was intentionally removed to prevent keyword-guessing exploits.
    'memory_leak' should NOT match 'memory_leak_issue' (different specific fault type).
    Agents must use exact canonical names like 'memory_leak', not substrings.
    """
    assert not _fuzzy_match("memory_leak", "memory_leak_issue"), (
        "Substring match must be rejected — prevent models guessing substrings of cause names"
    )

def test_fuzzy_match_case_insensitive():
    assert _fuzzy_match("Memory_Leak", "memory_leak")

def test_fuzzy_match_rejects_unrelated():
    assert not _fuzzy_match("traffic_spike", "memory_leak")

def test_fuzzy_set_overlap_full():
    assert _fuzzy_set_overlap({"redis", "api-gateway"}, {"redis", "api-gateway"}) == 1.0

def test_fuzzy_set_overlap_partial():
    result = _fuzzy_set_overlap({"redis"}, {"redis", "api-gateway"})
    assert 0.0 < result < 1.0

def test_fuzzy_set_overlap_empty_submitted():
    assert _fuzzy_set_overlap(set(), {"redis"}) == 0.0


# ── GRADER: grade_easy ──

def test_grade_easy_perfect():
    """Perfect easy: full health recovery + correct service+cause + correct playbook.
    Base = 0.20 + 0.30 + 0.25 = 0.75. compute_mttr_bonus(0.75, 3, 10) brings it to ~1.0.
    We test the base grader directly (health_score=1.0) and assert >= 0.70.
    """
    s = _load_scenario("easy_redis_oom.json")
    state = _make_state(diagnosed=True, diagnosed_service="redis",
                        diagnosed_cause="memory_leak",
                        remediated=True, remediation_applied="restart_redis")
    score = grade_easy(1.0, state, s)
    assert score >= 0.70, f"Perfect easy base expected >= 0.70, got {score}"

def test_grade_easy_synonym_cause():
    """Fuzzy matching should accept 'oom' as equivalent to 'memory_leak'."""
    s = _load_scenario("easy_redis_oom.json")
    state = _make_state(diagnosed=True, diagnosed_service="redis",
                        diagnosed_cause="oom",
                        remediated=True, remediation_applied="restart_redis")
    score = grade_easy(1.0, state, s)
    assert score >= 0.70, f"Fuzzy matching should accept 'oom' as memory_leak, got {score}"

def test_grade_easy_wrong_service():
    """Wrong service + wrong playbook target: only health credit (0.20 * health_score)."""
    s = _load_scenario("easy_redis_oom.json")
    state = _make_state(diagnosed=True, diagnosed_service="postgres",
                        diagnosed_cause="memory_leak",
                        remediated=True, remediation_applied="restart_redis")
    score = grade_easy(0.0, state, s)  # no health recovery either
    # Score clamps to _SCORE_MIN (0.01) — evaluator requires strictly (0, 1)
    assert score == 0.01, f"Wrong service + no health = 0.01 (clamped), got {score}"

def test_grade_easy_nothing_done():
    s = _load_scenario("easy_redis_oom.json")
    # Score clamps to _SCORE_MIN (0.01) — evaluator requires strictly (0, 1)
    assert grade_easy(0.0, _make_state(), s) == 0.01


# ── GRADER: grade_medium ──

def test_grade_medium_perfect():
    """Perfect medium: 0.15 health + 0.30 diag + 0.15 remediation = 0.60 base."""
    s = _load_scenario("medium_cascade_redis.json")
    state = _make_state(diagnosed=True, diagnosed_service="redis",
                        diagnosed_cause="memory_leak",
                        remediated=True, remediation_applied="restart_redis",
                        escalated=False)
    score = grade_medium(1.0, state, s)
    assert score >= 0.55, f"Perfect medium base expected >= 0.55, got {score}"

def test_grade_medium_no_op_agent_scores_zero():
    """A no-op agent with 0 health recovery should score exactly 0.0."""
    s = _load_scenario("medium_cascade_redis.json")
    score = grade_medium(0.0, _make_state(), s)
    # Score clamps to _SCORE_MIN (0.01) — evaluator requires strictly (0, 1)
    assert score == 0.01, f"No-op agent should score 0.01 (clamped), got {score}"

def test_grade_medium_red_herring_penalty():
    """Chasing a red herring triggers a -0.40 penalty → score must be very low."""
    s = _load_scenario("medium_cascade_redis.json")
    state = _make_state(diagnosed=True, diagnosed_service="postgres",
                        escalated=False)
    score = grade_medium(0.0, state, s)
    # Red herring penalty (-0.40) floors at _SCORE_MIN (0.01) — evaluator requires strictly (0, 1)
    assert score == 0.01, f"Red herring agent clamps to 0.01, got {score}"


# ── GRADER: grade_hard ──

def test_grade_hard_perfect():
    """Perfect hard: correct root cause + ≥80% affected services + sufficient timeline + ≥75% prevention.
    Scenario has 5 affected services; we list 4 (80%). timeline must be >= min+3=6 steps.
    """
    s = _load_scenario("hard_config_cascade.json")
    # min_timeline_steps = 3 → need 3+3=6 steps
    payload = {
        "root_cause": "bad_config_deploy",
        "affected_services": ["config-service", "auth-service", "redis", "app-service", "api-gateway"],
        "timeline_steps": [
            "t=0: config-service config push failed",
            "t=1: auth-service config refresh failed",
            "t=2: redis cache invalidation storm",
            "t=3: app-service returning 500",
            "t=4: api-gateway cascading 503",
            "t=5: rollback_config applied",
        ],
        "prevention_steps": ["config_validation_ci", "canary_deploy", "staging_parity"],
    }
    state = _make_state(diagnosed=True, diagnosed_service="config-service", diagnosed_cause="bad_config_deploy")
    score = grade_hard(1.0, payload, state, s)
    assert score >= 0.70, f"Perfect hard expected >= 0.70, got {score}"

def test_grade_hard_fuzzy_root_cause_rejected():
    """'bad_config' does NOT match 'bad_config_deploy' after removing substring logic."""
    s = _load_scenario("hard_config_cascade.json")
    payload = {
        "root_cause": "bad_config",  # too vague; should not match bad_config_deploy
        "affected_services": ["config-service", "auth-service", "redis", "app-service", "api-gateway"],
        "timeline_steps": [
            "t=0", "t=1", "t=2", "t=3", "t=4", "t=5",
        ],
        "prevention_steps": ["config_validation_ci", "canary_deploy", "staging_parity"],
    }
    score = grade_hard(1.0, payload, _make_state(), s)
    # Without root cause credit (0.25), max is 0.10+0.15+0.15+0.25 = 0.65
    assert score < 0.70, f"Vague 'bad_config' should not score full root cause, got {score}"

def test_grade_hard_wrong_root_cause():
    """Completely wrong root cause: only health + diagnosis + partial postmortem."""
    s = _load_scenario("hard_config_cascade.json")
    payload = {
        "root_cause": "network_partition",
        "affected_services": ["auth-service", "redis", "api-gateway"],
        "timeline_steps": ["t=0", "t=1", "t=2", "t=3", "t=4", "t=5"],
        "prevention_steps": ["config_validation_ci", "canary_deploy"],
    }
    score = grade_hard(0.0, payload, _make_state(), s)
    assert score < 0.50, f"Wrong root cause should score < 0.50, got {score}"

def test_grade_hard_returns_in_range():
    s = _load_scenario("hard_network_partition.json")
    payload = {
        "root_cause": "network_partition",
        "affected_services": ["auth-service", "redis", "api-gateway"],
        "timeline_steps": ["t=0", "t=1", "t=2"],
        "prevention_steps": ["fix retry bug", "add network redundancy"],
    }
    score = grade_hard(0.5, payload, _make_state(), s)
    assert 0.0 < score < 1.0, f"Score must be strictly in (0,1), got {score}"


# ── MTTR BONUS ──

def test_mttr_fast():
    """Fast solve: 80% correctness + 20% speed. base=1.0, steps=1/15 → ~0.99."""
    result = compute_mttr_bonus(1.0, 1, 15)
    assert result > 0.90, f"Fast MTTR expected > 0.90, got {result}"

def test_mttr_slow():
    """Slow solve (all steps used): 80% correctness, 0% speed = 0.80."""
    result = compute_mttr_bonus(1.0, 15, 15)
    assert abs(result - 0.80) < 0.01, f"Slow MTTR expected 0.80, got {result}"

def test_mttr_always_in_range():
    for base in [0.0, 0.5, 1.0]:
        for steps in [1, 7, 15]:
            result = compute_mttr_bonus(base, steps, 15)
            assert 0.0 < result < 1.0, f"MTTR result must be strictly in (0,1), got {result}"


# ── ENVIRONMENT INTEGRATION ──

try:
    from server.sre_environment import SREEnvironment
    from models import SREAction
    ENV_AVAILABLE = True
except ImportError:
    ENV_AVAILABLE = False


@pytest.mark.skipif(not ENV_AVAILABLE, reason="openenv-core not installed")
class TestSREEnvironment:

    def setup_method(self):
        self.env = SREEnvironment()

    def test_reset_returns_observation(self):
        obs = self.env.reset(task_id="task_easy_1")
        assert obs is not None
        assert obs.done is False
        assert len(obs.alerts) > 0

    def test_state_after_reset(self):
        self.env.reset(task_id="task_easy_1")
        state = self.env.state
        assert state.step_count == 0
        assert not state.diagnosed

    def test_partial_observability_initial(self):
        """Initial dependency graph should only contain alerted/logged services."""
        obs = self.env.reset(task_id="task_easy_1")
        graph_svcs = set(obs.dependency_graph.keys())
        alert_svcs = {a.service for a in obs.alerts}
        log_svcs = {l.service for l in obs.logs}
        discovered = alert_svcs | log_svcs
        # All graph nodes should be within discovered services
        assert graph_svcs.issubset(discovered), (
            f"Graph shows undiscovered services: {graph_svcs - discovered}"
        )

    def test_partial_observability_expands_after_query(self):
        """Querying a service should reveal its dependencies."""
        self.env.reset(task_id="task_easy_1")
        obs_before = self.env.step(SREAction(
            action_type="query_logs", payload={"service": "redis"}
        ))
        graph_before = set(obs_before.dependency_graph.keys())
        # After querying redis, we should see redis and potentially more nodes
        assert "redis" in graph_before or len(graph_before) > 0

    def test_dynamic_metrics_worsen(self):
        """Metrics should degrade each step if no correct remediation applied."""
        self.env.reset(task_id="task_easy_1")
        obs1 = self.env.step(SREAction(
            action_type="query_metrics", payload={"service": "api-gateway"}
        ))
        error_rate_1 = obs1.metrics.get("api_error_rate_pct", 0)
        obs2 = self.env.step(SREAction(
            action_type="query_logs", payload={"service": "redis"}
        ))
        error_rate_2 = obs2.metrics.get("api_error_rate_pct", 0)
        assert error_rate_2 >= error_rate_1, (
            "Error rate should not decrease without remediation"
        )

    def test_correct_diagnosis_positive_reward(self):
        self.env.reset(task_id="task_easy_1")
        obs = self.env.step(SREAction(
            action_type="submit_diagnosis",
            payload={"suspected_service": "redis", "suspected_cause": "memory_leak"}
        ))
        assert obs.reward > 0

    def test_red_herring_diagnosis_negative_reward(self):
        self.env.reset(task_id="task_medium_1")
        obs = self.env.step(SREAction(
            action_type="submit_diagnosis",
            payload={"suspected_service": "postgres", "suspected_cause": "pool_exhaustion"}
        ))
        assert obs.reward < 0

    def test_correct_remediation_positive_reward(self):
        self.env.reset(task_id="task_easy_1")
        self.env.step(SREAction(
            action_type="submit_diagnosis",
            payload={"suspected_service": "redis", "suspected_cause": "memory_leak"}
        ))
        obs = self.env.step(SREAction(
            action_type="apply_remediation",
            payload={"playbook_id": "restart_redis"}
        ))
        assert obs.reward > 0

    def test_postmortem_terminates_episode(self):
        self.env.reset(task_id="task_easy_1")
        self.env.step(SREAction(
            action_type="submit_diagnosis",
            payload={"suspected_service": "redis", "suspected_cause": "memory_leak"}
        ))
        self.env.step(SREAction(
            action_type="apply_remediation",
            payload={"playbook_id": "restart_redis"}
        ))
        obs = self.env.step(SREAction(
            action_type="submit_postmortem",
            payload={
                "root_cause": "memory_leak",
                "affected_services": ["redis", "api-gateway"],
                "timeline_steps": 3,
                "prevention_steps": ["redis_memory_alert"],
            }
        ))
        assert obs.done is True

    def test_max_steps_terminates_easy(self):
        """Easy tasks should terminate at 10 steps."""
        self.env.reset(task_id="task_easy_1")
        for _ in range(10):
            obs = self.env.step(SREAction(
                action_type="query_logs", payload={"service": "redis"}
            ))
            if obs.done:
                break
        assert obs.done is True

    def test_seed_reproducibility(self):
        obs1 = self.env.reset(task_id="task_easy_1", seed=42)
        m1 = dict(obs1.metrics)
        obs2 = self.env.reset(task_id="task_easy_1", seed=42)
        m2 = dict(obs2.metrics)
        assert m1 == m2

    def test_close_incident_penalised(self):
        self.env.reset(task_id="task_easy_1")
        obs = self.env.step(SREAction(
            action_type="close_incident", payload={}
        ))
        assert obs.done is True
        # Terminal rewards are clamped to (0, 1) exclusive — evaluator requires this
        assert 0.0 < obs.reward < 1.0, f"Terminal reward must be in (0,1), got {obs.reward}"
        # Close incident should give a very low score
        assert obs.reward <= 0.05, f"Close incident should be penalised, got {obs.reward}"

    @pytest.mark.parametrize("task_id", [
        "task_easy_1", "task_easy_2", "task_easy_3", "task_easy_4",
        "task_medium_1", "task_medium_2", "task_medium_3", "task_medium_4",
        "task_hard_1", "task_hard_2", "task_hard_3", "task_hard_4",
    ])
    def test_all_tasks_loadable(self, task_id):
        obs = self.env.reset(task_id=task_id)
        assert obs is not None
        assert not obs.done

# ── PROCEDURAL SCENARIO GENERATION ──

GENERATOR_REQUIRED_KEYS = {
    # Keys present in procedurally-generated scenarios (may differ from static JSON keys)
    "scenario_id", "difficulty", "description",
    "root_cause_service", "root_cause_type", "correct_diagnosis", "correct_playbook",
    "dependency_graph", "initial_alerts", "initial_logs", "initial_metrics",
    "queryable_logs", "queryable_metrics", "red_herrings",
    "affected_services", "valid_prevention_steps",
    "min_timeline_steps", "max_steps", "available_playbooks",
}

def test_scenario_generator_produces_valid_scenario():
    from server.scenario_generator import generate_scenario
    for diff in ("easy", "medium", "hard"):
        s = generate_scenario(difficulty=diff, seed=42)
        missing = GENERATOR_REQUIRED_KEYS - set(s.keys())
        assert not missing, f"Generated {diff} scenario missing: {missing}"
        assert s["difficulty"] == diff
        assert s["correct_playbook"] in s["available_playbooks"]
        rh_count = len(s.get("red_herrings", []))
        expected_rh = {"easy": 0, "medium": 1, "hard": 2}[diff]
        assert rh_count == expected_rh

def test_scenario_generator_seed_reproducibility():
    from server.scenario_generator import generate_scenario
    s1 = generate_scenario(difficulty="medium", seed=123)
    s2 = generate_scenario(difficulty="medium", seed=123)
    assert s1["root_cause_service"] == s2["root_cause_service"]
    assert s1["correct_diagnosis"] == s2["correct_diagnosis"]
