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


def _pick_machine(op, ready_ops):
    """
    Choose machine that minimizes completion time, balancing SPT with load.
    For each eligible machine, estimate when it would be free based on other
    ready operations that might also use it, then pick the machine giving
    earliest completion for this op.
    """
    mo = op["machine_options"]
    if not mo:
        return 0
    
    best_machine = None
    best_completion = float('inf')
    
    for m, d in mo.items():
        # Estimate load: count how many other ready ops prefer this machine
        # as their best (shortest) option
        load_factor = 1.0
        for other in ready_ops:
            if other is op:
                continue
            other_mo = other.get("machine_options", {})
            if m in other_mo:
                # If this machine is among the top-2 shortest for other,
                # it's more likely to be busy
                sorted_durs = sorted(other_mo.values())
                if len(sorted_durs) >= 2 and other_mo[m] <= sorted_durs[1]:
                    load_factor += 0.5
                elif len(sorted_durs) == 1 and other_mo[m] == sorted_durs[0]:
                    load_factor += 0.5
        
        # Effective completion time considering expected contention
        est_completion = d * load_factor
        
        if est_completion < best_completion:
            best_completion = est_completion
            best_machine = m
    
    return best_machine if best_machine is not None else 0


def _critical_path_length(op):
    """
    Estimate the longest remaining path (critical path) for this job.
    Uses remaining_work as a proxy, adjusted by the number of remaining ops
    to account for sequential dependencies.
    """
    rem_work = op.get("remaining_work", 0)
    rem_ops = op.get("remaining_ops", 0)
    
    if rem_ops <= 0:
        return 0.0
    
    # Critical path is at least the sum of minimum durations of remaining ops.
    # remaining_work approximates this. Scale by 1/rem_ops to get per-op avg,
    # but we want total path length, so use remaining_work directly.
    # Add a small bonus for jobs with more remaining ops (longer critical path
    # in terms of number of sequential steps).
    cp = rem_work
    # Jobs with many remaining ops have longer critical paths; weight by
    # sqrt(rem_ops) to emphasize without over-amplifying.
    cp += 0.2 * np.sqrt(max(rem_ops, 1))
    
    return cp


def _score_operation(op, time, ready_ops):
    """
    Composite score: higher is better (we pick max).
    Primary: critical path length (longest remaining path) to prevent idle time
    Secondary: remaining work (urgency)
    Tertiary: shortest processing time (SPT) to reduce machine idle time
    """
    rem_work = op.get("remaining_work", 0)
    rem_ops = op.get("remaining_ops", 0)
    mo = op.get("machine_options", {})

    if mo:
        min_proc = min(mo.values())
    else:
        min_proc = 0

    # Critical path: highest priority
    cp = _critical_path_length(op)
    
    # Composite score
    score = 1.0 * cp          # critical path length (primary)
    score += 0.5 * rem_work   # remaining work (secondary urgency)
    score += 0.1 * rem_ops    # more remaining ops = longer chain, keep flowing
    score -= 0.3 * min_proc   # prefer shorter processing times (SPT)

    return score


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: if we've used >110s, return immediately with a fast heuristic
    if _elapsed() > 110.0:
        # Fast fallback: just pick max remaining_work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Compute composite scores
    scores = [_score_operation(op, time, ready_operations) for op in ready_operations]
    best_idx = int(np.argmax(scores))

    # For the chosen operation, pick the best machine considering load
    chosen_op = ready_operations[best_idx]
    machine = _pick_machine(chosen_op, ready_operations)

    return (best_idx, machine)
# EVOLVE-BLOCK-END
