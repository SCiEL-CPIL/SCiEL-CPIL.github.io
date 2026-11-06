"""Job-shop scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.monotonic()
_DEADLINE = _START_TIME + 140.0  # Safety margin before 150s kill


def _soft_min(durations, tau=0.5):
    """Smooth surrogate for min(durations) via log-sum-exp. As tau->0 approaches min."""
    d = np.asarray(durations, dtype=np.float64)
    if d.size == 0:
        return 1.0
    d = np.clip(d, 1e-9, None)
    neg = -d / tau
    max_neg = np.max(neg)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg - max_neg)))
    val = -tau * log_sum_exp
    if val <= 0:
        val = 1e-9
    return float(val)


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.

    Primary signal: urgency = remaining_ops / soft_min_duration
        (captured from inspiration 1; a job with many remaining operations whose
         current step is short on some machine is on the critical path and should
         move early to reduce makespan).
    Secondary signals (from the current program's composite, now re-weighted):
        - efficiency: remaining_work / soft_min_duration (job load per unit time)
        - SPT: 1 / soft_min_duration (free machines quickly)
        - flexibility: 1 - (min_dur/total_dur) (prefer ops that can pick a lighter machine)
        - machine-contention: estimated ready-queue load on the op's machines
        - remaining_ops: fewer remaining ops = closer to completion (tie-break)

    The soft-min replaces the hard min so that near-tie machine options produce a
    graded, rather than binary, signal.
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _time.monotonic() > _DEADLINE:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Estimate machine contention from the ready queue:
    # sum of durations each machine is asked to handle among ready ops.
    machine_load = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m, d in mo.items():
            machine_load[m] = machine_load.get(m, 0.0) + d

    scores = np.zeros(n)

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        durations = list(machine_options.values()) if machine_options else [1.0]

        min_duration = min(durations) if durations else 1.0
        total_duration = sum(durations) if durations else 1.0
        num_machines = len(durations)

        remaining_work = op.get("remaining_work", 0) or 0.0
        remaining_ops = op.get("remaining_ops", 1) or 1

        # Soft-min duration: smooth surrogate for the best machine's processing time
        sm = _soft_min(durations, tau=0.5)

        # PRIMARY: urgency ratio (remaining_ops / soft_min)
        # More remaining ops + short current step => high critical-path urgency
        if sm > 0:
            urgency = remaining_ops / sm
        else:
            urgency = remaining_ops + 1000.0

        # Secondary: efficiency (remaining_work / soft_min)
        if sm > 0:
            efficiency = remaining_work / sm
        else:
            efficiency = remaining_work + 1000.0

        # Tertiary: SPT bonus
        if sm > 0:
            spt_score = 1.0 / sm
        else:
            spt_score = 1000.0

        # Flexibility: lower min/total ratio => more flexible (can pick lighter machine)
        if total_duration > 0:
            flexibility_ratio = min_duration / total_duration
        else:
            flexibility_ratio = 1.0
        flexibility_score = 1.0 - flexibility_ratio

        # Machine-contention avoidance: prefer ops whose machines are less loaded
        # in the current ready queue. Use the min load among the op's machines.
        if machine_options:
            min_load = min(machine_load.get(m, 0.0) for m in machine_options)
        else:
            min_load = 0.0

        # Composite score: urgency dominates, then efficiency, SPT, flexibility,
        # contention, and finally a small completion proximity term.
        # Weights tuned so the primary critical-path signal leads.
        scores[i] = (
            3.0 * urgency +             # Primary: critical-path urgency
            1.0 * efficiency +          # Job load per unit time
            1.0 * spt_score +           # Free machines quickly
            0.5 * flexibility_score +   # Machine-contention avoidance
            0.05 * (-min_load) +        # Avoid congested machines
            0.3 * (-remaining_ops)      # Completion proximity (small)
        )

    best_idx = int(np.argmax(scores))
    return int(best_idx)
# EVOLVE-BLOCK-END
