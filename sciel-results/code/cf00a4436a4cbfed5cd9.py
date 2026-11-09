"""Job-Shop Scheduling Dispatch Rule.
Uses a heuristic combining Criticality (Remaining Work), 
Efficiency (Shortest Processing Time), and Urgency (Fewest Remaining Ops).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TS = _time.time()
DEADLINE_S = 110.0  # Safety margin against 120s/150s limit

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options, key=machine_options.get)
            return (0, best_machine)
        return 0

    # Deadline check: fallback to LPT (max min-duration) if time is running out
    if _time.time() - _START_TS > DEADLINE_S:
        min_durs = []
        for op in ready_operations:
            mo = op.get("machine_options", {})
            if mo:
                min_durs.append(min(mo.values()))
            else:
                min_durs.append(0.0)
        idx = int(np.argmax(min_durs))
        op = ready_operations[idx]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options, key=machine_options.get)
            return (idx, best_machine)
        return idx

    # Extract features
    remaining_works = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op.get("remaining_ops", 1) for op in ready_operations], dtype=np.float64)
    
    # Calculate min duration and best machine for each operation
    min_durations = np.zeros(n, dtype=np.float64)
    best_machines = [None] * n
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_m, min_d = min(machine_options.items(), key=lambda item: item[1])
            min_durations[i] = min_d
            best_machines[i] = best_m
        else:
            min_durations[i] = 0.0
            best_machines[i] = None

    # LPT (Greedy Bottleneck Elimination) structural heuristic:
    # Prioritize operations with the LONGEST minimum processing time.
    # Rationale: Long operations are inflexible and likely to become critical path
    # bottlenecks. Scheduling them early reduces the chance they delay the makespan.
    #
    # Secondary tie-breakers:
    # 1. Higher remaining_work (more critical job)
    # 2. Fewer remaining_ops (closer to completion)
    
    # Primary: LPT score (higher min_duration is better)
    max_min_dur = np.max(min_durations) if np.max(min_durations) > 0 else 1.0
    lpt_score = min_durations / max_min_dur  # [0, 1], 1 = longest
    
    # Secondary: remaining work (higher is more critical)
    max_work = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    work_score = remaining_works / max_work  # [0, 1]
    
    # Tertiary: remaining ops (fewer = closer to done)
    remaining_ops_safe = np.maximum(remaining_ops, 1.0)
    max_ops = np.max(remaining_ops_safe) if np.max(remaining_ops_safe) > 0 else 1.0
    ops_score = (max_ops - remaining_ops_safe) / max_ops  # [0, 1], 1 = fewest
    
    # Weighted combination with LPT dominant
    w_lpt = 0.7
    w_work = 0.2
    w_ops = 0.1
    
    scores = (w_lpt * lpt_score) + (w_work * work_score) + (w_ops * ops_score)
    
    best_idx = int(np.argmax(scores))
    
    best_machine = best_machines[best_idx]
    
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx

# EVOLVE-BLOCK-END
