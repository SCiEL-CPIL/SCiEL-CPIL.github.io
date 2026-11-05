"""Open-Shop Scheduling dispatch rule using a machine-load balancing strategy."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE = 120.0
_SAFETY_MARGIN = 5.0


def _elapsed() -> float:
    return _time.time() - _START_TIME


def _fallback_dispatch(ready_operations):
    """
    Fast fallback heuristic if deadline is approaching.
    Prioritizes max remaining_work, then min duration.
    """
    best_idx = 0
    best_rw = -1
    best_dur = float('inf')
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        durations = op["machine_options"]
        if not durations:
            continue
        min_d = min(durations.values())
        
        # Primary: remaining work, Secondary: min duration
        if rw > best_rw or (rw == best_rw and min_d < best_dur):
            best_rw = rw
            best_dur = min_d
            best_idx = i
            
    op = ready_operations[best_idx]
    durations = op["machine_options"]
    if durations:
        best_m = min(durations, key=lambda m: durations[m])
    else:
        best_m = 0
    return (best_idx, best_m)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    
    Returns (index, machine) tuple.
    
    Strategy: Machine-load balancing.
    We prioritize operations that target the most constrained machines (highest contention).
    By scheduling operations on busy machines early, we prevent them from becoming
    the dominant bottleneck. Among operations targeting a specific machine, we prefer
    those with shorter durations (SPT) to clear the queue quickly, and higher remaining
    work to handle critical jobs.
    """
    n = len(ready_operations)
    if n == 0:
        return 0  # safe default; should not happen
    
    # Check deadline guard
    if _elapsed() > _DEADLINE - _SAFETY_MARGIN:
        return _fallback_dispatch(ready_operations)

    if n == 1:
        op = ready_operations[0]
        durations = op["machine_options"]
        if durations:
            # For a single op, pick the machine with the highest contention?
            # No, if only one op is ready, it doesn't matter for contention.
            # Pick min duration to finish fast.
            best_m = min(durations, key=lambda m: durations[m])
        else:
            best_m = 0
        return (0, best_m)

    # 1. Calculate contention for each machine
    # Contention is the number of ready operations that can use this machine.
    machine_contention = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_contention[m] = machine_contention.get(m, 0) + 1

    # 2. Score each operation
    # We want to maximize the score.
    # Components:
    # A. Machine Constraint: How "congested" is the best machine for this op?
    #    We define the "best machine" for an op as the one with the highest contention
    #    among its options, because that is the bottleneck we want to address.
    #    Alternatively, we can look at the machine with the *lowest* effective cost.
    #    Let's stick to the idea: identify the most constrained machine this op can hit.
    #    If an op can go to Machine A (cont=5) or B (cont=1), it's a candidate for A.
    #    We score it higher if it can address a high-contention machine.
    
    # B. Duration: Shorter is better (SPT) to clear the bottleneck fast.
    # C. Remaining Work: Higher is better (LRPT) to handle critical jobs.
    
    # Let's refine the score:
    # For each op, find the machine m* that maximizes (Contention[m] / Duration[m])?
    # Or simply:
    # Score = w1 * Max_Contention_Available - w2 * Min_Duration + w3 * Remaining_Work
    # This feels a bit disjointed.
    
    # Better approach:
    # For each op, evaluate its "value" on each machine it can use.
    # Value(m) = (Contention[m] * 100) / (Duration[m] + 1)
    # This rewards using busy machines with short durations.
    # Then take the max Value across available machines for that op.
    # Add a small weight for Remaining Work to break ties or handle criticality.
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = 0
    
    # Weights
    W_CONTENTION = 1.0
    W_WORK = 0.1  # Lower weight for work to prioritize load balancing, but keep it
    
    for i, op in enumerate(ready_operations):
        rw = float(op["remaining_work"])
        durations = op["machine_options"]
        
        if not durations:
            continue

        max_machine_value = -float('inf')
        best_m_for_op = 0
        
        for m, d in durations.items():
            cont = machine_contention.get(m, 0)
            # Value of assigning this op to machine m
            # High contention is good (we are balancing load by addressing it)
            # Low duration is good (we clear it fast)
            # If d is very small, value is high.
            # We use d in denominator to penalize long durations.
            # To avoid division by zero, add 1.
            machine_value = (W_CONTENTION * cont) / (d + 1.0)
            
            if machine_value > max_machine_value:
                max_machine_value = machine_value
                best_m_for_op = m
        
        # Combine with remaining work
        # We want to maximize.
        # Score = max_machine_value + W_WORK * rw
        # Note: rw can be large, so we might need to normalize or adjust weights.
        # In open shop, rw is typically sum of remaining durations.
        # If rw is ~50 and cont/d is ~1-5, W_WORK=0.1 makes rw contribute ~5.
        # This seems reasonable.
        
        score = max_machine_value + W_WORK * rw
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
