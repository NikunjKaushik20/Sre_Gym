
import re
from difflib import SequenceMatcher

# ── Score clamping — evaluator requires strictly (0, 1) exclusive ─────────────
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99

def _clamp_score(x: float) -> float:
    """Clamp a score to (_SCORE_MIN, _SCORE_MAX) and round to 4 decimal places."""
    return round(max(_SCORE_MIN, min(float(x), _SCORE_MAX)), 4)

# ── Strict synonym map — ONLY exact canonical variants, no loose aliases ──────
# Intentionally narrow: "config" alone does NOT match "bad_config_deploy"
_SYNONYMS = {
    "memory_leak":          {"oom", "out_of_memory", "memory_exhaustion"},
    "traffic_spike":        {"traffic_flood"},
    "jwt_bug":              {"jwt_validation_bug", "token_validation_bug"},
    "bad_config_deploy":    {"bad_config_deploy", "config_deploy_error"},
    "deploy_regression":    {"deployment_regression"},
    "dns_misconfiguration": {"dns_misconfiguration"},
    "disk_full":            {"disk_exhaustion", "no_disk_space"},
    "config_typo":          {"env_var_typo"},
    "pool_exhaustion":      {"connection_pool_exhaustion"},
    "cert_expiry":          {"certificate_expiry", "tls_expiry"},
    "network_partition":    {"network_split", "network_isolation"},
}


def _norm(s):
    if not isinstance(s, str) or not s:
        return ""
    return re.sub(r"[\s\-]+", "_", s.strip().lower())


def _fuzzy(submitted, expected, threshold=0.92):
    """Strict fuzzy match.

    Rules (in order):
      1. Exact normalised match → True
      2. Strict synonym lookup (both tokens must map to same canonical) → True
      3. SequenceMatcher ratio ≥ threshold (default 0.92) → True
      4. Everything else → False

    Critically: NO substring matching (s in e / e in s) — this was the
    primary exploit allowing models to submit "config" and match
    "bad_config_deploy".
    """
    s, e = _norm(submitted), _norm(expected)
    if not s or not e:
        return False
    if s == e:
        return True
    # Strict synonym lookup
    for canon, syns in _SYNONYMS.items():
        canon_n = _norm(canon)
        forms = {canon_n} | {_norm(x) for x in syns}
        if e in forms and s in forms:
            return True
    # High-bar SequenceMatcher (requires very close strings)
    return SequenceMatcher(None, s, e).ratio() >= threshold


def _set_overlap(submitted, expected):
    """Fraction of expected items matched in submitted (strict fuzzy)."""
    if not expected:
        return 1.0
    if not submitted:
        return 0.0
    return sum(
        1 for e in expected if any(_fuzzy(s, e) for s in submitted)
    ) / len(expected)


# ── EASY ──────────────────────────────────────────────────────────────────────

def grade_easy(health_score, state, scenario):
    """
    Easy scoring (max 1.0):
      20% health recovery       — strictly capped; health alone cannot carry
      30% correct diagnosis     — BOTH service AND cause required (strict match)
      25% correct remediation   — only awarded if diagnosis is fully correct
      25% reserved for MTTR     — applied by compute_mttr_bonus caller

    Key tightenings vs. previous version:
      - Health weight: 0.35 → 0.20
      - Diagnosis weight: 0.25 → 0.30 (but now requires cause match too)
      - Service-only partial credit: 0.05 → 0.0 (no partial on service alone)
      - Wrong remediation without diagnosis: 0.05 → 0.0
    """
    r = min(health_score, 1.0) * 0.20

    diag_correct = False
    if state.diagnosed:
        service_ok = _fuzzy(str(state.diagnosed_service), scenario["fault_service"])
        cause_ok   = _fuzzy(str(state.diagnosed_cause),   scenario.get("fault_type", ""))
        if service_ok and cause_ok:
            r += 0.30
            diag_correct = True
        # No partial credit for service-only or cause-only guesses

    if state.remediated:
        effects = scenario.get("playbook_effects", {})
        pb      = str(state.remediation_applied)
        if pb in effects and effects[pb]["target"] == scenario["fault_service"]:
            if diag_correct:
                r += 0.25
            # Applying the right fix without correct diagnosis = 0 credit

    return _clamp_score(r)


# ── MEDIUM ────────────────────────────────────────────────────────────────────

def grade_medium(health_score, state, scenario):
    """
    Medium scoring (max 1.0):
      15% health recovery       — minimal; not a free 30 points
      30% correct diagnosis     — service AND cause, strict match
      15% correct remediation   — gated on full correct diagnosis
      40% reserved for MTTR     — applied by compute_mttr_bonus caller

    Penalties:
      Red herring hit:  -0.40 (was -0.35; now crosses into negative range)
      Wrong service:    -0.15 (was -0.10)
      Service-only hit:  +0.0 (was +0.05 — removed partial credit)

    The 40% MTTR pool means a wrong-but-fast agent still fails badly.
    """
    r = min(health_score, 1.0) * 0.15

    rh_services = [x["service"] for x in scenario.get("red_herrings", [])]

    if state.diagnosed:
        service_ok = _fuzzy(str(state.diagnosed_service), scenario["fault_service"])
        cause_ok   = _fuzzy(str(state.diagnosed_cause),   scenario.get("fault_type", ""))

        if state.diagnosed_service in rh_services:
            r -= 0.40          # chased a red herring — now floors the score heavily
        elif service_ok and cause_ok:
            r += 0.30          # correct service + correct cause (strict)
        elif service_ok:
            r -= 0.05          # found service but missed cause = slight penalty
        else:
            r -= 0.15          # wrong service, not a red herring

    if state.remediated:
        effects = scenario.get("playbook_effects", {})
        pb      = str(state.remediation_applied)
        if pb in effects and effects[pb]["target"] == scenario["fault_service"]:
            service_ok = state.diagnosed and _fuzzy(
                str(state.diagnosed_service), scenario["fault_service"]
            )
            cause_ok = state.diagnosed and _fuzzy(
                str(state.diagnosed_cause), scenario.get("fault_type", "")
            )
            if service_ok and cause_ok:
                r += 0.15
            # Lucky fix without diagnosis = 0 (was +0.03 — removed)

    return _clamp_score(r)


# ── HARD ──────────────────────────────────────────────────────────────────────

def grade_hard(health_score, postmortem, state, scenario):
    """
    Hard scoring (max 1.0):
      10% health recovery       — almost irrelevant alone
      10% correct diagnosis     — service identification, strict match
      80% postmortem quality    — 5 sub-components, all with raised bars:
          25% root cause        (strict threshold 0.92, no partial at 0.70)
          15% affected services (≥0.80 overlap required, was 0.70)
          15% timeline depth    (min_timeline_steps + 3, was + 2)
          25% prevention steps  (≥0.75 overlap required, was 0.65)
           0% if postmortem missing — no credit for empty submission

    Red herring in diagnosis: -0.20 (was -0.15).
    Wrong service (non-red-herring): -0.05 applied.

    Key tightenings:
      - Health weight 0.20 → 0.10
      - Root cause partial at 0.70 → removed entirely (binary)
      - Affected services threshold 0.70 → 0.80
      - Prevention threshold 0.65 → 0.75
      - Timeline bar raised by +1
      - Prevention partial (0.40–0.65) → 0 (removed; was +0.05)
    """
    r = min(health_score, 1.0) * 0.10

    rh_services = [x["service"] for x in scenario.get("red_herrings", [])]

    # Diagnosis component
    if state.diagnosed:
        if state.diagnosed_service in rh_services:
            r -= 0.20
        elif _fuzzy(str(state.diagnosed_service), scenario["fault_service"]):
            r += 0.10
        else:
            r -= 0.05   # wrong service penalty

    # Guard: if postmortem is empty or not a dict, skip all sub-scores
    if not isinstance(postmortem, dict) or not postmortem:
        return _clamp_score(r)

    # ── Postmortem sub-scores ──────────────────────────────────────────

    # 1. Root cause — strict threshold 0.92, binary (no partial anymore)
    root_cause_text = str(postmortem.get("root_cause", ""))
    if _fuzzy(root_cause_text, scenario.get("fault_type", ""), threshold=0.92):
        r += 0.25
    # No partial credit at lower thresholds — guessing must be precise

    # 2. Affected services — require ≥0.80 overlap (raised from 0.70)
    expected_aff = set(scenario.get("affected_services", scenario.get("services", [])))
    sub_aff      = set(postmortem.get("affected_services", []))
    aff_overlap  = _set_overlap(sub_aff, expected_aff)
    if aff_overlap >= 0.80:
        r += 0.15 * aff_overlap
    # Below 0.80 = zero (listing one service in a five-service cascade is noise)

    # 3. Timeline depth — scenario minimum + 3 steps (raised from +2)
    ts  = postmortem.get("timeline_steps", 0)
    num = len(ts) if isinstance(ts, list) else int(ts) if isinstance(ts, (int, float)) else 0
    required = scenario.get("min_timeline_steps", 2) + 3
    if num >= required:
        r += 0.15
    elif num >= scenario.get("min_timeline_steps", 2) + 2:
        r += 0.05   # narrower partial band (met last bar, missed new bar)

    # 4. Prevention steps — require ≥0.75 overlap; proportional above that
    valid_prev  = set(scenario.get("valid_prevention_steps", []))
    sub_prev    = set(postmortem.get("prevention_steps", []))
    prev_overlap = _set_overlap(sub_prev, valid_prev)
    if prev_overlap >= 0.75:
        r += 0.25 * prev_overlap
    # below 0.75 = zero (was 0.40 partial floor — removed)

    return _clamp_score(r)


# ── MTTR BONUS ────────────────────────────────────────────────────────────────

def compute_mttr_bonus(base, steps, max_steps):
    """
    Final score = 80% correctness + 20% speed efficiency.

    Tightened from 75/25 to 80/20 split so that speed cannot compensate
    for a weak base score. A model that guesses randomly quickly should
    not be rewarded.

    An agent that scores base=0.50 but solves in 3/15 steps:
      = (0.50 * 0.80) + ((1 - 3/15) * 0.20) = 0.40 + 0.160 = 0.560

    An agent that scores base=0.50 but uses all 15 steps:
      = (0.50 * 0.80) + (0 * 0.20) = 0.40

    Contrast old formula (75/25):
      = (0.50 * 0.75) + (0.80 * 0.25) = 0.375 + 0.20 = 0.575  ← inflated
    """
    eff = max(0.0, 1.0 - steps / max(1, max_steps))
    return _clamp_score((base * 0.80) + (eff * 0.20))


# ── PUBLIC ALIASES (keep test suite compatible) ───────────────────────────────
_fuzzy_match        = _fuzzy
_fuzzy_set_overlap  = _set_overlap