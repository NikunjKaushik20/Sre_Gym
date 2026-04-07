"""
Pytest test suite for SRE-Gym.

Tests verify:
  - Grader functions return correct [0.0, 1.0] values
  - Environment reset / step / state flow
  - Reward shaping (correct diagnosis = positive, red herring = negative)
  - Payload validation catches missing fields
  - MTTR bonus formula
  - All 12 scenario files load without error
  - Termination on max_steps
  - Dynamic metrics update after correct remediation
"""
import json
import sys
import pytest
from pathlib import Path
from types import SimpleNamespace

# Add sre_gym to path for local imports
SRE_GYM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SRE_GYM_ROOT))
sys.path.insert(0, str(SRE_GYM_ROOT / "server"))

from server.graders import grade_easy, grade_medium, grade_hard, compute_mttr_bonus

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

SCENARIOS_DIR = SRE_GYM_ROOT / "scenarios"

ALL_SCENARIO_FILES = [
    "easy_redis_oom.json",
    "easy_postgres_slow.json",
    "easy_config_typo.json",
    "easy_disk_full.json",
    "medium_cascade_redis.json",
    "medium_cascade_auth.json",
    "medium_traffic_spike.json",
    "medium_cert_expiry.json",
    "hard_config_cascade.json",
    "hard_network_partition.json",
    "hard_deploy_regression.json",
    "hard_multi_region.json",
]

REQUIRED_SCENARIO_KEYS = {
    "scenario_id", "task_id", "difficulty",
    "root_cause_service", "correct_diagnosis", "correct_playbook",
    "dependency_graph", "initial_alerts", "initial_logs", "initial_metrics",
    "queryable_logs", "queryable_metrics", "red_herrings",
    "affected_services", "valid_prevention_steps",
    "min_timeline_steps", "max_steps", "available_playbooks",
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


def _load_scenario(filename: str) -> dict:
    with open(SCENARIOS_DIR / filename) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# SCENARIO FILE INTEGRITY
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("filename", ALL_SCENARIO_FILES)
def test_scenario_file_exists(filename):
    assert (SCENARIOS_DIR / filename).exists(), f"Missing scenario file: {filename}"


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
def test_scenario_root_cause_service_in_graph(filename):
    scenario = _load_scenario(filename)
    graph = scenario["dependency_graph"]
    all_services = set(graph.keys()) | {s for deps in graph.values() for s in deps}
    rcs = scenario["root_cause_service"]
    assert rcs in all_services, (
        f"{filename}: root_cause_service '{rcs}' not in dependency graph"
    )


@pytest.mark.parametrize("filename", ALL_SCENARIO_FILES)
def test_scenario_correct_playbook_in_available(filename):
    scenario = _load_scenario(filename)
    assert scenario["correct_playbook"] in scenario["available_playbooks"], (
        f"{filename}: correct_playbook '{scenario['correct_playbook']}' "
        f"not in available_playbooks"
    )


@pytest.mark.parametrize("filename", ALL_SCENARIO_FILES)
def test_scenario_red_herrings_not_root_cause(filename):
    scenario = _load_scenario(filename)
    rh_services = {rh["service"] for rh in scenario.get("red_herrings", [])}
    assert scenario["root_cause_service"] not in rh_services, (
        f"{filename}: root_cause_service appears in red_herrings"
    )


@pytest.mark.parametrize("filename", [f for f in ALL_SCENARIO_FILES if "hard" in f])
def test_hard_scenario_has_two_red_herrings(filename):
    scenario = _load_scenario(filename)
    count = len(scenario.get("red_herrings", []))
    assert count >= 2, f"{filename}: hard scenario should have >=2 red herrings, found {count}"


@pytest.mark.parametrize("filename", [f for f in ALL_SCENARIO_FILES if "medium" in f])
def test_medium_scenario_has_one_red_herring(filename):
    scenario = _load_scenario(filename)
    count = len(scenario.get("red_herrings", []))
    assert count >= 1, f"{filename}: medium scenario should have >=1 red herrings, found {count}"


# ──────────────────────────────────────────────────────────────────────────────
# GRADER: grade_easy
# ──────────────────────────────────────────────────────────────────────────────

def test_grade_easy_perfect_score():
    scenario = _load_scenario("easy_redis_oom.json")
    state = _make_state(
        diagnosed=True,
        diagnosed_service="redis",
        diagnosed_cause="memory_leak",
        remediated=True,
        remediation_applied="restart_redis",
    )
    score = grade_easy(state, scenario)
    assert score == 1.0


def test_grade_easy_wrong_service():
    scenario = _load_scenario("easy_redis_oom.json")
    state = _make_state(
        diagnosed=True,
        diagnosed_service="postgres",   # wrong
        diagnosed_cause="memory_leak",
        remediated=True,
        remediation_applied="restart_redis",
    )
    score = grade_easy(state, scenario)
    assert score < 0.5, f"Expected score < 0.5 for wrong service, got {score}"


def test_grade_easy_correct_service_wrong_cause():
    scenario = _load_scenario("easy_redis_oom.json")
    state = _make_state(
        diagnosed=True,
        diagnosed_service="redis",
        diagnosed_cause="wrong_cause",   # wrong cause
        remediated=True,
        remediation_applied="restart_redis",
    )
    score = grade_easy(state, scenario)
    # Service OK + remediation OK = 0.4+0.4 = 0.8 (cause missed = -0.2)
    assert 0.7 <= score <= 0.9


def test_grade_easy_wrong_playbook():
    scenario = _load_scenario("easy_redis_oom.json")
    state = _make_state(
        diagnosed=True,
        diagnosed_service="redis",
        diagnosed_cause="memory_leak",
        remediated=True,
        remediation_applied="scale_postgres_pool",  # wrong
    )
    score = grade_easy(state, scenario)
    assert score < 0.7


def test_grade_easy_nothing_done():
    scenario = _load_scenario("easy_redis_oom.json")
    state = _make_state()
    score = grade_easy(state, scenario)
    assert score == 0.0


def test_grade_easy_returns_float_in_range():
    scenario = _load_scenario("easy_redis_oom.json")
    state = _make_state(
        diagnosed=True, diagnosed_service="redis", diagnosed_cause="memory_leak",
        remediated=True, remediation_applied="restart_redis",
    )
    score = grade_easy(state, scenario)
    assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# GRADER: grade_medium
# ──────────────────────────────────────────────────────────────────────────────

def test_grade_medium_perfect_score():
    scenario = _load_scenario("medium_cascade_redis.json")
    state = _make_state(
        diagnosed=True,
        diagnosed_service="redis",
        diagnosed_cause="memory_leak",
        remediated=True,
        remediation_applied="restart_redis",
        escalated=False,
    )
    score = grade_medium(state, scenario)
    assert score == 1.0


def test_grade_medium_red_herring_penalty():
    scenario = _load_scenario("medium_cascade_redis.json")
    # postgres is the red herring
    state = _make_state(
        diagnosed=True,
        diagnosed_service="postgres",   # red herring!
        diagnosed_cause="memory_leak",
        remediated=False,
        escalated=False,
    )
    score = grade_medium(state, scenario)
    assert score < 0.3, f"Expected heavy penalty for red herring diagnosis, got {score}"


def test_grade_medium_unnecessary_escalation_penalty():
    scenario = _load_scenario("medium_cascade_redis.json")
    # correct_escalation_team is null for this scenario
    state = _make_state(
        diagnosed=True,
        diagnosed_service="redis",
        diagnosed_cause="memory_leak",
        remediated=True,
        remediation_applied="restart_redis",
        escalated=True,
        escalation_team="database",  # unnecessary escalation
    )
    score = grade_medium(state, scenario)
    assert score < 1.0, "Unnecessary escalation should reduce score"


def test_grade_medium_returns_float_in_range():
    scenario = _load_scenario("medium_cascade_auth.json")
    state = _make_state(
        diagnosed=True, diagnosed_service="auth-service", diagnosed_cause="jwt_bug",
        remediated=True, remediation_applied="rollback_config", escalated=False,
    )
    score = grade_medium(state, scenario)
    assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# GRADER: grade_hard
# ──────────────────────────────────────────────────────────────────────────────

def test_grade_hard_perfect_postmortem():
    scenario = _load_scenario("hard_config_cascade.json")
    state = _make_state(
        diagnosed=True,
        diagnosed_service="auth-service",
        diagnosed_cause="bad_config_deploy",
    )
    payload = {
        "root_cause":        "bad_config_deploy",
        "affected_services": ["auth-service", "redis", "api-gateway"],
        "timeline_steps":    4,
        "prevention_steps":  ["test config in staging", "canary deploy"],
    }
    score = grade_hard(payload, state, scenario)
    assert score == 1.0


def test_grade_hard_wrong_root_cause():
    scenario = _load_scenario("hard_config_cascade.json")
    state = _make_state()
    payload = {
        "root_cause":        "network_issue",   # wrong
        "affected_services": ["auth-service", "redis", "api-gateway"],
        "timeline_steps":    4,
        "prevention_steps":  ["test config in staging", "canary deploy"],
    }
    score = grade_hard(payload, state, scenario)
    assert score < 0.7, "Wrong root cause should significantly reduce score"


def test_grade_hard_missing_affected_services():
    scenario = _load_scenario("hard_config_cascade.json")
    state = _make_state()
    payload = {
        "root_cause":        "bad_config_deploy",
        "affected_services": [],    # none submitted
        "timeline_steps":    4,
        "prevention_steps":  ["test config in staging", "canary deploy"],
    }
    score = grade_hard(payload, state, scenario)
    assert score < 0.8, "Missing affected services should reduce score"


def test_grade_hard_partial_prevention_steps():
    scenario = _load_scenario("hard_config_cascade.json")
    state = _make_state()
    payload = {
        "root_cause":        "bad_config_deploy",
        "affected_services": ["auth-service", "redis", "api-gateway"],
        "timeline_steps":    4,
        "prevention_steps":  ["test config in staging"],   # only 1 of 2
    }
    score = grade_hard(payload, state, scenario)
    # Should be < 1.0 but > 0.7
    assert 0.7 < score < 1.0


def test_grade_hard_returns_float_in_range():
    scenario = _load_scenario("hard_network_partition.json")
    state = _make_state()
    payload = {
        "root_cause":        "network_partition",
        "affected_services": ["auth-service", "redis", "api-gateway"],
        "timeline_steps":    3,
        "prevention_steps":  ["fix retry bug", "add network redundancy"],
    }
    score = grade_hard(payload, state, scenario)
    assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# GRADER: compute_mttr_bonus
# ──────────────────────────────────────────────────────────────────────────────

def test_mttr_bonus_fast_resolution():
    """Resolving in 1 step with perfect score should approach 1.0."""
    final = compute_mttr_bonus(base_reward=1.0, steps_taken=1, max_steps=15)
    assert final > 0.9


def test_mttr_bonus_slow_resolution():
    """Resolving at max steps with perfect score should yield 75% of base."""
    final = compute_mttr_bonus(base_reward=1.0, steps_taken=15, max_steps=15)
    assert abs(final - 0.75) < 0.01


def test_mttr_bonus_zero_base():
    final = compute_mttr_bonus(base_reward=0.0, steps_taken=5, max_steps=15)
    assert final >= 0.0


def test_mttr_bonus_always_in_range():
    for base in [0.0, 0.5, 1.0]:
        for steps in [1, 7, 15]:
            result = compute_mttr_bonus(base, steps, 15)
            assert 0.0 <= result <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT: Integration Tests
# ──────────────────────────────────────────────────────────────────────────────

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
        assert len(obs.dependency_graph) > 0

    def test_state_after_reset(self):
        self.env.reset(task_id="task_easy_1")
        state = self.env.state
        assert state.step_count == 0
        assert not state.diagnosed
        assert not state.remediated

    def test_step_query_logs(self):
        self.env.reset(task_id="task_easy_1")
        obs = self.env.step(SREAction(
            action_type="query_logs",
            payload={"service": "redis"}
        ))
        assert obs is not None
        assert obs.done is False
        assert obs.reward is not None

    def test_correct_diagnosis_positive_reward(self):
        self.env.reset(task_id="task_easy_1")  # redis OOM
        obs = self.env.step(SREAction(
            action_type="submit_diagnosis",
            payload={"suspected_service": "redis", "suspected_cause": "memory_leak"}
        ))
        assert obs.reward > 0, "Correct diagnosis should yield positive reward"

    def test_red_herring_diagnosis_negative_reward(self):
        self.env.reset(task_id="task_medium_1")  # postgres is red herring
        obs = self.env.step(SREAction(
            action_type="submit_diagnosis",
            payload={"suspected_service": "postgres", "suspected_cause": "pool_exhaustion"}
        ))
        assert obs.reward < 0, "Red herring diagnosis should yield negative reward"

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

    def test_wrong_remediation_negative_reward(self):
        self.env.reset(task_id="task_easy_1")
        obs = self.env.step(SREAction(
            action_type="apply_remediation",
            payload={"playbook_id": "scale_postgres_pool"}   # wrong
        ))
        assert obs.reward < 0

    def test_missing_payload_fields_penalised(self):
        self.env.reset(task_id="task_easy_1")
        obs = self.env.step(SREAction(
            action_type="submit_diagnosis",
            payload={}   # missing suspected_service and suspected_cause
        ))
        assert obs.reward < 0
        assert obs.done is False

    def test_invalid_action_type_penalised(self):
        self.env.reset(task_id="task_easy_1")
        obs = self.env.step(SREAction(
            action_type="hack_the_mainframe",
            payload={}
        ))
        assert obs.reward <= 0
        assert obs.done is False

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
                "root_cause":        "memory_leak",
                "affected_services": ["redis", "api-gateway"],
                "timeline_steps":    3,
                "prevention_steps":  ["redis_memory_alert"],
            }
        ))
        assert obs.done is True

    def test_max_steps_terminates_episode(self):
        self.env.reset(task_id="task_easy_1")
        scenario_max_steps = 15
        obs = None
        for _ in range(scenario_max_steps):
            obs = self.env.step(SREAction(
                action_type="query_logs",
                payload={"service": "redis"}
            ))
            if obs.done:
                break
        assert obs is not None and obs.done is True

    def test_seed_reproducibility(self):
        obs1 = self.env.reset(task_id="task_easy_1", seed=42)
        m1   = dict(obs1.metrics)
        obs2 = self.env.reset(task_id="task_easy_1", seed=42)
        m2   = dict(obs2.metrics)
        assert m1 == m2, "Same seed should produce identical metrics"

    def test_seed_variation(self):
        obs1 = self.env.reset(task_id="task_easy_1", seed=1)
        m1   = dict(obs1.metrics)
        obs2 = self.env.reset(task_id="task_easy_1", seed=999)
        m2   = dict(obs2.metrics)
        # Metrics will differ (within ±5%) — at least one value should differ
        assert m1 != m2, "Different seeds should produce different metric values"

    def test_dynamic_metrics_after_correct_remediation(self):
        self.env.reset(task_id="task_easy_1")
        obs_before = self.env.step(SREAction(
            action_type="query_metrics",
            payload={"service": "redis"}
        ))
        error_rate_before = obs_before.metrics.get("api_error_rate_pct", 0)
        self.env.step(SREAction(
            action_type="apply_remediation",
            payload={"playbook_id": "restart_redis"}
        ))
        obs_after = self.env.step(SREAction(
            action_type="query_metrics",
            payload={"service": "api-gateway"}
        ))
        error_rate_after = obs_after.metrics.get("api_error_rate_pct", 0)
        assert error_rate_after < error_rate_before, (
            "Error rate should drop after correct remediation"
        )

    def test_close_incident_without_resolution_penalised(self):
        self.env.reset(task_id="task_easy_1")
        obs = self.env.step(SREAction(
            action_type="close_incident",
            payload={}
        ))
        assert obs.done is True
        assert obs.reward <= 0.0

    @pytest.mark.parametrize("task_id", list({
        "task_easy_1", "task_easy_2", "task_easy_3", "task_easy_4",
        "task_medium_1", "task_medium_2", "task_medium_3", "task_medium_4",
        "task_hard_1", "task_hard_2", "task_hard_3", "task_hard_4",
    }))
    def test_all_tasks_loadable(self, task_id):
        obs = self.env.reset(task_id=task_id)
        assert obs is not None
        assert not obs.done
        assert len(obs.alerts) > 0
