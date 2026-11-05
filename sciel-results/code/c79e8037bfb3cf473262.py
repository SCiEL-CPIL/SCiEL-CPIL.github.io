"""Open-Shop Scheduling dispatch rule for open-shop event-driven simulation."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE_MARGIN = 5  # seconds before the 150s kill; be conservative


def _elapsed() -> float:
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    1. Primary: Job with most remaining work (LRPT) to prioritize critical path.
    2. Secondary: Operation with longer duration on best machine (LPT) to clear critical items.
    3. Machine: Machine with shortest duration for the chosen operation (SPT).
    
    Return: (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        best_m, best_d = _best_machine_for_op(op)
        return (0, best_m)

    # Deadline guard: if we are too close to the limit, return a quick heuristic
    if _elapsed() > 145:
        # Quick fallback: pick first op, min duration machine
        op = ready_operations[0]
        best_m, _ = _best_machine_for_op(op)
        return (0, best_m)

    scores = []
    best_machines = []
    
    for op in ready_operations:
        rw = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        durations = op["machine_options"]
        if not durations:
            scores.append(-1)
            best_machines.append(None)
            continue
            
        min_dur = min(durations.values())
        max_dur = max(durations.values())
        
        # Smoother scoring: combine remaining work (LRPT) with operation criticality
        # Use a geometric-mean-like blend to avoid abrupt threshold effects
        # Primary: remaining_work dominates
        # Secondary: longer min_dur indicates more critical operation (LPT within job)
        # Tertiary: spread between min and max dur indicates machine flexibility
        # Use log-sum-exp style smoothing for the min_dur component
        if rw > 0:
            # Normalize duration relative to remaining work for scale invariance
            dur_ratio = min_dur / (rw + 1e-9)
            # Blend: heavy weight on rw, moderate on dur_ratio for differentiation
            score = rw * 100.0 + dur_ratio * 10.0 + (max_dur - min_dur) * 0.1
        else:
            score = min_dur
        
        scores.append(score)
        
        best_m, _ = _best_machine_for_op(op)
        best_machines.append(best_m)

    # Find index with max score
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    if best_machine is None:
        return best_idx
    return (best_idx, best_machine)


def _best_machine_for_op(op):
    """
    For a given operation, select the machine that minimizes its duration.
    """
    durations = op["machine_options"]
    if not durations:
        return None, 0
    best_machine = min(durations, key=lambda m: durations[m])
    return best_machine, durations[best_machine]
# EVOLVE-BLOCK-END
