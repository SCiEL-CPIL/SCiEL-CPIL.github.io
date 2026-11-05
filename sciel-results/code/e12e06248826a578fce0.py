"""Open-Shop Scheduling dispatch rule using a contention-aware scoring approach."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE = 120.0
_SAFETY_MARGIN = 5.0


def _elapsed() -> float:
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    
    Returns (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Deadline guard
    if _elapsed() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: Max remaining work, Min duration
        best_idx = 0
        best_rw = -1
        for i, op in enumerate(ready_operations):
            rw = op["remaining_work"]
            if rw > best_rw:
                best_rw = rw
                best_idx = i
        op = ready_operations[best_idx]
        durations = op["machine_options"]
        best_m = min(durations, key=lambda m: durations[m]) if durations else 0
        return (best_idx, best_m)

    if n == 1:
        op = ready_operations[0]
        durations = op["machine_options"]
        best_m = min(durations, key=lambda m: durations[m]) if durations else 0
        return (0, best_m)

    # 1. Calculate contention for each machine
    # Contention is the number of ready operations that can use this machine.
    machine_contention = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_contention[m] = machine_contention.get(m, 0) + 1

    # Parameters
    # We reward remaining work (criticality) and max duration (MPT/LPT for bottleneck reduction).
    # We penalize the effective cost (duration * contention) to avoid clogging busy machines.
    W_RW = 1.0       # weight of remaining work
    W_REM = 0.5      # weight of remaining ops
    W_MAX = 0.8      # weight of max duration (rewarding long ops)
    W_COST = 1.0     # weight of min_cost penalty
    BETA = 0.6       # weight of contention factor in cost

    best_score = -float('inf')
    best_idx = 0
    best_machine = 0

    for i, op in enumerate(ready_operations):
        rw = float(op["remaining_work"])
        rem_ops = float(op.get("remaining_ops", 1))
        durations = op["machine_options"]
        
        if not durations:
            continue

        durs = list(durations.values())
        max_dur = max(durs)
        
        # Calculate cost for each machine: duration * (1 + beta * contention)
        costs = {}
        for m, d in durations.items():
            cont = machine_contention.get(m, 0)
            costs[m] = d * (1.0 + BETA * cont)
        
        # Find the best machine (min cost)
        best_m_for_op = min(costs, key=lambda m: costs[m])
        min_cost_val = costs[best_m_for_op]
        
        # Calculate score
        # High rw, high rem_ops, high max_dur are good.
        # Low min_cost is good.
        score = W_RW * rw + W_REM * rem_ops + W_MAX * max_dur - W_COST * min_cost_val
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
