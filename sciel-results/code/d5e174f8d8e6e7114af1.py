"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = None

def _ensure_init():
    global _dispatch_start_time
    if _dispatch_start_time is None:
        _dispatch_start_time = _time.time()

def _time_left():
    if _dispatch_start_time is None:
        return 115.0
    return 115.0 - (_time.time() - _dispatch_start_time)

def dispatch(ready_operations, time):
    _ensure_init()
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If we're running low on time, use the fastest heuristic
    if _time_left() < 5.0:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Compute scores for each ready operation
    # Strategy: Combine multiple heuristics for open-shop scheduling
    # Key insight: In open-shop, we want to minimize makespan.
    # The critical path is determined by the longest job (sum of all ops) and 
    # the busiest machine (sum of all ops on that machine).
    # A good heuristic is to prioritize operations that are on the critical path.
    
    scores = []
    # Build a machine load proxy: for each machine, sum the min durations
    # of all ready ops that can use it. This approximates which machines
    # are most constrained in the current ready set.
    machine_load = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            if m not in machine_load:
                machine_load[m] = 0.0
            machine_load[m] += d
    
    max_machine_load = max(machine_load.values()) if machine_load else 1.0
    if max_machine_load < 1e-9:
        max_machine_load = 1.0
    
    for op in ready_operations:
        work = op["remaining_work"]
        durations = list(op["machine_options"].values())
        if not durations:
            scores.append(0.0)
            continue
        
        min_dur = min(durations)
        max_dur = max(durations)
        avg_dur = sum(durations) / len(durations)
        num_machines = len(durations)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Primary: Most Work Remaining (MWR)
        # This is the dominant factor in open-shop scheduling
        score = work
        
        # Secondary: Prefer operations with shorter processing times
        # to keep machines busy and reduce idle time
        # Use min duration as a proxy for "quick to complete"
        score += 0.1 * (1.0 / (min_dur + 1e-9))
        
        # Tertiary: Prefer operations that are closer to completing their job
        # Completing a job frees it from consideration
        if remaining_ops > 0:
            score += 0.05 * (1.0 / remaining_ops)
        
        # Quaternary: Slight penalty for very long operations
        # to avoid blocking a machine for too long
        score -= 0.02 * max_dur
        
        # Quinary: Prefer operations with more machine options (flexibility)
        score += 0.01 * num_machines
        
        # New: Machine-load balancing term.
        # For each machine option of this op, compute how "loaded" that machine
        # is relative to the max. Prefer ops that can be assigned to less-loaded
        # machines. Use the min normalized load across options as a proxy for
        # the best available machine for this op.
        if machine_load:
            # Normalized load per machine option: lower = less loaded
            norm_loads = [machine_load[m] / max_machine_load for m in op["machine_options"]]
            best_load = min(norm_loads)  # 0.0 = completely free, 1.0 = most loaded
            # Reward ops that can go to less-loaded machines
            score += 0.15 * (1.0 - best_load)
        
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
