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
    # Lambda: weight of the "effective duration" penalty.
    # Beta: weight of the contention factor.
    # Gamma: temperature for soft-min (lower = closer to hard min)
    LAMBDA = 1.2
    BETA = 0.7
    GAMMA = 2.0
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = 0

    for i, op in enumerate(ready_operations):
        rw = float(op["remaining_work"])
        durations = op["machine_options"]
        
        if not durations:
            continue

        # Compute soft-min of costs using log-sum-exp for smoothness
        costs = []
        machines = []
        for m, d in durations.items():
            cont = machine_contention.get(m, 0)
            cost = d * (1.0 + BETA * cont)
            costs.append(cost)
            machines.append(m)
        
        # Soft-min: -log(sum(exp(-cost/gamma))) * gamma
        # For numerical stability, subtract max cost first
        costs_arr = np.array(costs, dtype=np.float64)
        max_cost = np.max(costs_arr)
        # exp(-(cost - max_cost)/gamma) to avoid overflow
        exp_terms = np.exp(-(costs_arr - max_cost) / GAMMA)
        log_sum_exp = np.log(np.sum(exp_terms)) + (max_cost / GAMMA)
        soft_min_cost = -GAMMA * log_sum_exp + max_cost
        
        # Find the machine with the lowest actual cost for tie-breaking
        min_cost_idx = int(np.argmin(costs_arr))
        best_m_for_op = machines[min_cost_idx]
        
        # Calculate score for this operation
        # Score = remaining_work - LAMBDA * soft_min_cost
        # This balances criticality (remaining work) against the smooth "cost" of scheduling it
        score = rw - LAMBDA * soft_min_cost
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
