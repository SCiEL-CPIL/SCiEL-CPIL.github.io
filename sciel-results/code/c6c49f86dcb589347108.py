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


def _critical_path_length(op):
    """
    Estimate the critical path length for this operation.
    Uses remaining_work as a proxy for the total processing time still needed
    along the critical path from this op to job completion.
    """
    return op.get("remaining_work", 0)


def _score_operation(op, time):
    """
    Composite score: higher is better (we pick max).
    Balances:
      - critical_path_length (urgency: ops on critical path should start early)
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

    cp_len = _critical_path_length(op)

    # Weight critical path length heavily to prioritize ops on the critical path
    # Weight remaining_ops to favor completing jobs (fewer ops left = more urgent)
    # Weight min_proc inversely (SPT to reduce machine idle time)
    
    score = 1.0 * cp_len
    score += 0.2 * (rem_ops if rem_ops > 0 else 1)
    score -= 0.3 * min_proc

    return score


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: if we've used >110s, return immediately with a fast heuristic
    if _elapsed() > 110.0:
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
