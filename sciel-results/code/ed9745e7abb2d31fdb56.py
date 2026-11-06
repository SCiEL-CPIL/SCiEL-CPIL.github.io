"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: keep explicit wall-clock budget.
    # The evaluator kills after 150s; we want to return well before that.
    _start = _time.perf_counter()
    _deadline = _start + 115.0  # 115s safety margin

    # Structural Change: 
    # Instead of static weights, we use a "Greedy Load Balancing" heuristic.
    # We estimate the current "load" of each machine based on the ready set.
    # Then, for each candidate operation, we calculate a cost that reflects:
    # 1. How much this operation adds to the critical path (duration).
    # 2. How busy the target machine is (to avoid bottlenecking).
    # 3. A small tie-breaker for job depth (closer to completion).
    
    # First, estimate machine loads from the current ready operations.
    # This is a proxy for "which machines are about to get congested".
    # We sum the minimum required durations for each machine across all ready ops.
    # Note: This is a static snapshot, but it captures the immediate pressure.
    machine_loads = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m, d in mo.items():
            if m not in machine_loads:
                machine_loads[m] = 0.0
            # We add the duration to the load. 
            # This assumes that if multiple ops want the same machine, 
            # the total time they will occupy it is a good proxy for congestion.
            machine_loads[m] += d

    best_score = float('inf')
    best_index = 0

    for i, op in enumerate(ready_operations):
        # Deadline guard
        if _time.perf_counter() > _deadline:
            break

        remaining_work = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})

        if not machine_options:
            # Fallback: if no machine options, just score high (bad) or low?
            # Usually shouldn't happen. If it does, treat as very bad.
            score = float('inf')
        else:
            # Find the best machine for this operation in terms of "cost"
            min_cost = float('inf')
            
            for m, d in machine_options.items():
                if d <= 0:
                    d = 1e-9
                
                # Load of this machine based on other ready ops (approximation)
                # We use the pre-computed load. 
                # A higher load means this machine is more congested.
                load = machine_loads.get(m, 0.0)
                
                # Heuristic:
                # Primary: Minimize the time this op takes on this machine (SPT).
                # Secondary: Avoid machines that are already heavily loaded (Balance).
                # The "cost" is the processing time weighted by the machine's congestion.
                # We use a soft penalty: cost = d * (1 + load_factor)
                # load_factor is normalized to [0, 1] roughly to keep scale sane.
                # Since we don't know the total time scale, we use a simple ratio.
                # Let's just use d + load * 0.1 as a simple additive penalty.
                # Or better: d * (1 + load / (d + 1)) to make it relative.
                
                # Simple effective cost:
                # If machine is very loaded, the effective wait + process is higher.
                # We approximate effective time as d + (load * 0.2)
                # This encourages spreading load.
                
                effective_time = d + load * 0.2
                
                # Also consider the depth: jobs closer to completion might be 
                # slightly more urgent to free up the job, but in makespan minimization,
                # SPT and Load Balancing are usually dominant.
                # Let's add a small bonus for jobs with fewer remaining ops.
                depth_bonus = 1.0 / (remaining_ops + 1)
                
                # We want to MINIMIZE cost.
                # Score = effective_time - depth_bonus (since higher depth_bonus is better, subtract it)
                cost = effective_time - depth_bonus
                
                if cost < min_cost:
                    min_cost = cost

            score = min_cost

        if score < best_score:
            best_score = score
            best_index = i

    return int(best_index)
# EVOLVE-BLOCK-END
