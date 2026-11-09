"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = _time.time()
    return _time.time() - _START


def _pick_machine(op):
    """Choose the machine with the shortest processing time for this op."""
    mo = op["machine_options"]
    if not mo:
        return 0
    best_machine = None
    best_dur = float('inf')
    for m, d in mo.items():
        if d < best_dur:
            best_dur = d
            best_machine = m
    return best_machine if best_machine is not None else 0


def _score_operation(op, time):
    """
    Composite score: higher is better (we pick max).
    Balances:
      - remaining_work (urgency: long jobs should start early)
      - remaining_ops (fewer remaining ops = closer to completion, reduce WIP)
      - min processing time (SPT on chosen machine reduces idle time)
    """
    rem_work = op.get("remaining_work", 0)
    rem_ops = op.get("remaining_ops", 0)
    mo = op.get("machine_options", {})

    if mo:
        min_proc = min(mo.values())
    else:
        min_proc = 0

    # Normalize-ish: use raw values with tuned weights.
    # Weight remaining_work heavily to avoid late start of long jobs.
    # Weight remaining_ops inversely (fewer ops left = more urgent to finish).
    # Weight min_proc inversely (SPT to reduce machine idle time).
    
    score = 1.0 * rem_work
    score += 0.3 * (rem_ops if rem_ops > 0 else 1)  # more remaining ops = higher priority to keep flowing
    score -= 0.5 * min_proc  # prefer shorter processing times

    return score


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: if we've used >100s, return immediately with a fast heuristic
    if _elapsed() > 100.0:
        # Fast fallback: just pick max remaining_work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Compute composite scores
    scores = [_score_operation(op, time) for op in ready_operations]
    best_idx = int(np.argmax(scores))

    # For the chosen operation, pick the best machine (SPT)
    chosen_op = ready_operations[best_idx]
    machine = _pick_machine(chosen_op)

    return (best_idx, machine)
# EVOLVE-BLOCK-END
