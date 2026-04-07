"""Deterministic graders for SRE-Gym tasks."""

def grade_easy(state, scenario):
    """Grade easy task: correct diagnosis + correct remediation."""
    reward = 0.0
    if state.diagnosed and state.diagnosed_service == scenario["root_cause_service"]:
        reward += 0.4
        if state.diagnosed_cause == scenario["correct_diagnosis"]:
            reward += 0.2
    if state.remediated and state.remediation_applied == scenario["correct_playbook"]:
        reward += 0.4
    return min(reward, 1.0)

def grade_medium(state, scenario):
    """Grade medium task: diagnosis + root cause + remediation + no wrong escalation."""
    reward = 0.0
    if state.diagnosed_service == scenario["root_cause_service"]:
        reward += 0.25
        if state.diagnosed_cause == scenario["correct_diagnosis"]:
            reward += 0.25
    if state.remediation_applied == scenario["correct_playbook"]:
        reward += 0.25
    # Penalize diagnosing a red herring
    rh_services = [rh["service"] for rh in scenario.get("red_herrings", [])]
    if state.diagnosed_service in rh_services:
        reward -= 0.2
    # Bonus for not escalating unnecessarily
    if not state.escalated or state.escalation_team == scenario.get("correct_escalation_team"):
        reward += 0.25
    return max(0.0, min(reward, 1.0))

def grade_hard(postmortem_payload, state, scenario):
    """Grade hard task: full postmortem quality."""
    reward = 0.0
    if postmortem_payload.get("root_cause") == scenario["correct_diagnosis"]:
        reward += 0.35
    expected = set(scenario["affected_services"])
    submitted = set(postmortem_payload.get("affected_services", []))
    if expected:
        if not submitted:
            reward -= 0.1
        else:
            reward += 0.20 * (len(expected & submitted) / len(expected))
    else:
        reward += 0.20
    # Safely extract timeline_steps whether it's an int, or an LLM hallucinated list of strings
    timeline_steps = postmortem_payload.get("timeline_steps", 0)
    if isinstance(timeline_steps, list):
        num_steps = len(timeline_steps)
    elif isinstance(timeline_steps, (int, float)):
        num_steps = int(timeline_steps)
    else:
        try:
            num_steps = int(timeline_steps)
        except (ValueError, TypeError):
            num_steps = 0

    if num_steps >= scenario["min_timeline_steps"]:
        reward += 0.20
    valid = set(scenario["valid_prevention_steps"])
    submitted_prev = set(postmortem_payload.get("prevention_steps", []))
    reward += 0.25 * min(1.0, len(valid & submitted_prev) / max(len(valid), 1))
    return max(0.0, min(round(reward, 3), 1.0))

def compute_mttr_bonus(base_reward, steps_taken, max_steps):
    """MTTR efficiency multiplier — rewards faster resolution."""
    efficiency = max(0.0, 1.0 - (steps_taken / max_steps))
    final = (base_reward * 0.75) + (efficiency * 0.25)
    return round(max(0.0, min(final, 1.0)), 3)
