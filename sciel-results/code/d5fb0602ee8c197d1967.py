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
    
    # First pass: compute total remaining workload per machine across all ready operations
    # This helps identify which machines are under-loaded vs over-loaded
    machine_workload = {}
    for op in ready_operations:
        for machine, dur in op["machine_options"].items():
            if machine not in machine_workload:
                machine_workload[machine] = 0.0
            machine_workload[machine] += dur
    
    # Compute mean and std of machine workloads for normalization
    if machine_workload:
        workloads = np.array(list(machine_workload.values()))
        mean_wl = np.mean(workloads)
        std_wl = np.std(workloads) + 1e-9
    else:
        mean_wl = 0.0
        std_wl = 1.0
    
    scores = []
    for op in ready_operations:
        work = op["remaining_work"]
        durations = list(op["machine_options"].values())
        machines = list(op["machine_options"].keys())
        if not durations:
            scores.append(0.0)
            continue
        
        min_dur = min(durations)
        max_dur = max(durations)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Find the machine with the least remaining workload (most available)
        # that can process this operation
        min_machine_wl = float('inf')
        for m in machines:
            wl = machine_workload.get(m, 0.0)
            if wl < min_machine_wl:
                min_machine_wl = wl
        
        # Normalize the machine workload relative to mean
        # Lower workload machines get higher scores (they have more capacity)
        machine_balance_score = (mean_wl - min_machine_wl) / std_wl
        
        # Primary: Most Work Remaining (MWR) - jobs with more remaining work are more critical
        score = work
        
        # Secondary: Strong preference for shorter current operation (SPT-like)
        # Shorter operations finish sooner, freeing up resources
        score += 2.0 * (1.0 / (min_dur + 1.0))
        
        # Tertiary: Preference for jobs closer to completion
        if remaining_ops > 0:
            score += 0.5 * (1.0 / (remaining_ops + 1.0))
        
        # Quaternary: Machine balance - prioritize operations on less-loaded machines
        # This directly addresses the critical path by balancing machine utilization
        score += 1.5 * machine_balance_score
        
        # Quinary: Small penalty for very long operations on the worst case
        score -= 0.01 * max_dur
        
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
