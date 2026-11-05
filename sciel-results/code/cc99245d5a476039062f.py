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

    # 2. Calculate total remaining workload per machine
    # Sum of durations for all ready operations that can use this machine.
    # This acts as a proxy for the machine's current "load" in the bipartite graph.
    machine_workload = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            machine_workload[m] = machine_workload.get(m, 0) + d

    # Parameters
    # Lambda: weight of the "effective duration" penalty.
    # Beta: weight of the contention factor (count-based).
    # Gamma: weight of the workload factor (sum-based).
    # Alpha: weight of the remaining work (criticality).
    LAMBDA = 0.9
    BETA = 0.6
    GAMMA = 0.01  # Scale factor for workload (durations can be larger than counts)
    ALPHA = 1.0

    best_score = -float('inf')
    best_idx = 0
    best_machine = 0

    for i, op in enumerate(ready_operations):
        rw = float(op["remaining_work"])
        durations = op["machine_options"]
        
        if not durations:
            continue

        # Find the best machine for this operation based on effective cost
        # Cost = duration * (1 + beta * contention + gamma * workload)
        # We want to minimize this cost.
        min_cost = float('inf')
        best_m_for_op = 0
        
        for m, d in durations.items():
            cont = machine_contention.get(m, 0)
            work = machine_workload.get(m, 0)
            # Combine contention (number of competing ops) and workload (total time)
            # Workload is scaled down because it's a sum of durations, not a count
            cost = d * (1.0 + BETA * cont + GAMMA * work)
            if cost < min_cost:
                min_cost = cost
                best_m_for_op = m

        # Calculate score for this operation
        # Score = ALPHA * remaining_work - LAMBDA * min_cost
        # This balances criticality (remaining work) against the "cost" of scheduling it
        # (which is higher if it takes long on a busy machine).
        score = ALPHA * rw - LAMBDA * min_cost
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
