"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random

_START_TIME = None

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Safety guard: if we are approaching the 120s limit, fall back to a fast heuristic
    # to ensure we don't timeout. The deadline is 120s, we check against 115s.
    if _time.time() - _START_TIME > 115:
        # Fast fallback: pick the operation with the shortest duration on its preferred machine
        # to keep the system moving.
        min_dur = float('inf')
        best_idx = 0
        for i, op in enumerate(ready_operations):
            min_op_dur = min(op["machine_options"].values())
            if min_op_dur < min_dur:
                min_dur = min_op_dur
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    if n_ops == 1:
        return 0

    # 1. Estimate machine load based on the current ready queue.
    # A simple proxy for "machine congestion" is the sum of durations of all ready ops
    # that can run on that machine.
    machine_load = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            if m not in machine_load:
                machine_load[m] = 0
            machine_load[m] += d

    # 2. Compute a composite score for each operation:
    #    - remaining_work: prioritize jobs with more work left (longest job first)
    #    - machine_balance: prefer operations that can run on less-loaded machines
    #    - urgency: operations with fewer remaining ops are closer to completion
    #    
    # Score = alpha * remaining_work + beta * min_machine_load_ratio - gamma * remaining_ops
    # Where min_machine_load_ratio = min over machines of (1 / (machine_load[m] + duration))
    # This rewards picking ops that can run on lightly-loaded machines.
    
    alpha = 1.0
    beta = 0.5
    gamma = 0.3
    
    best_score = -float('inf')
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Find the best machine for this operation in terms of load balance
        # We want to minimize the impact on the most loaded machine
        best_machine_score = 0
        for m, d in op["machine_options"].items():
            load = machine_load.get(m, 0)
            # Higher score if machine has less load relative to the duration
            # Use 1/(load + d) as a proxy for "availability"
            if load + d > 0:
                availability = 1.0 / (load + d)
            else:
                availability = 0
            if availability > best_machine_score:
                best_machine_score = availability
        
        # Composite score
        score = alpha * rw + beta * best_machine_score * rw - gamma * remaining_ops
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
