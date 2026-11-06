"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time_mod
import numpy as np
from collections import defaultdict

_DEADLINE = _time_mod.monotonic() + 145.0  # safety margin before 150s kill

# Track machine loads across dispatch calls for load balancing
_machine_loads = defaultdict(float)
_dispatch_count = 0


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next in an open-shop simulation.
    
    Strategy:
    1. Select the job with the most remaining work (MWR) to reduce makespan.
    2. Among its ready operations, pick the one that minimizes a combined score
       of job priority and machine contention (load-balanced SPT).
    3. Use a smooth scoring function to balance job work and machine load.
    """
    global _machine_loads, _dispatch_count
    
    # Deadline guard
    if _time_mod.monotonic() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    _dispatch_count += 1
    
    # Pre-compute job remaining work
    job_work = {}
    for i, op in enumerate(ready_operations):
        j = op["job"]
        if j not in job_work:
            job_work[j] = op["remaining_work"]
    
    # Find max remaining work for normalization
    max_work = max(job_work.values()) if job_work else 1.0
    if max_work == 0:
        max_work = 1.0
    
    # Compute machine loads from current ready operations
    # Estimate load as sum of min durations for operations that could use each machine
    machine_estimated_load = defaultdict(float)
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if isinstance(machine_options, dict):
            for mach, dur in machine_options.items():
                machine_estimated_load[mach] += dur
        else:
            for mach in machine_options:
                machine_estimated_load[mach] += 1.0  # unknown duration
    
    # Combine tracked loads with estimated loads
    total_machine_load = {}
    all_machines = set(machine_estimated_load.keys()) | set(_machine_loads.keys())
    for mach in all_machines:
        est = machine_estimated_load.get(mach, 0.0)
        tracked = _machine_loads.get(mach, 0.0)
        total_machine_load[mach] = tracked + est
    
    max_load = max(total_machine_load.values()) if total_machine_load else 1.0
    if max_load == 0:
        max_load = 1.0
    
    # Score each ready operation
    best_idx = 0
    best_score = -float('inf')
    
    for i, op in enumerate(ready_operations):
        j = op["job"]
        work = op["remaining_work"]
        
        # Get machine options and their durations
        machine_options = op["machine_options"]
        if isinstance(machine_options, dict):
            items = list(machine_options.items())
        else:
            items = [(mach, 1.0) for mach in machine_options]
        
        if not items:
            continue
        
        # Find the best machine for this operation
        # Score: high job priority, low machine load, short duration
        best_machine_score = -float('inf')
        best_machine = None
        best_duration = float('inf')
        
        for mach, dur in items:
            if dur <= 0:
                continue
            # Normalize job work (higher is better)
            work_norm = work / max_work
            # Normalize machine load (lower is better)
            load_norm = total_machine_load.get(mach, 0.0) / max_load
            # Normalize duration (lower is better)
            dur_norm = dur / max_load if max_load > 0 else dur
            
            # Combined score: prioritize high work, then low load, then short duration
            # Use weighted combination with smooth terms
            score = 0.6 * work_norm - 0.25 * load_norm - 0.15 * (dur / max_load if max_load > 0 else dur)
            
            if score > best_machine_score:
                best_machine_score = score
                best_machine = mach
                best_duration = dur
        
        if best_machine is None:
            continue
        
        # Update tracked machine load (simulating that this operation will start)
        # We'll tentatively add the duration to the machine load
        _machine_loads[best_machine] += best_duration
        
        # Final score for this operation
        # Higher job work is better, lower resulting machine load is better
        final_score = best_machine_score
        
        if final_score > best_score:
            best_score = final_score
            best_idx = i
    
    # If no valid operation found, fall back to first
    if best_score == -float('inf'):
        best_idx = 0
        # Reset the tentative load addition
        if _machine_loads:
            # We don't know which machine was tentatively added, so we'll just leave it
            pass
    
    return int(best_idx)
# EVOLVE-BLOCK-END
