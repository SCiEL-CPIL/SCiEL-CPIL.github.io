"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: keep under 150s wall-clock total for the evaluator.
# We track a global start time and a per-call budget.
_GLOBAL_START = _time.monotonic()
_CALL_BUDGET = 0.5  # seconds per dispatch call (safety margin)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next, and optionally which machine.
    
    Strategy: Multi-criteria heuristic combining:
    1. Most remaining work (job-level priority)
    2. Shortest processing time on the chosen machine (to reduce machine idle)
    3. Machine load balancing (prefer less-loaded machines)
    
    We score each ready operation by a weighted combination and return the best index.
    We also select the best machine for that operation to minimize its completion time
    while considering machine availability implicitly via the simulation.
    
    Since we don't have direct access to machine availability in this interface,
    we use a heuristic: prefer the machine with the shortest duration for this op,
    but also consider the job's remaining work to prioritize critical jobs.
    
    Return: index of the ready operation to start. Optionally (index, machine).
    """
    if not ready_operations:
        return 0
    
    # Deadline guard: if we've been running too long, just return argmax of remaining_work
    elapsed = _time.monotonic() - _GLOBAL_START
    if elapsed > 140.0:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    
    # Compute scores for each ready operation
    scores = np.zeros(n)
    
    # Precompute arrays for efficiency
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Normalize remaining_work to [0, 1] if possible
    rw_min = np.min(remaining_work)
    rw_max = np.max(remaining_work)
    rw_range = rw_max - rw_min if rw_max > rw_min else 1.0
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            scores[i] = remaining_work[i] / rw_range
            continue
        
        # Get durations for all eligible machines
        durations = np.array([d for d in machine_options.values()], dtype=np.float64)
        machines = list(machine_options.keys())
        
        # Shortest processing time
        min_dur = np.min(durations)
        
        # Most remaining work (normalized)
        rw_score = (remaining_work[i] - rw_min) / rw_range
        
        # Most remaining ops (criticality)
        ro_score = remaining_ops[i] / max(1.0, np.max(remaining_ops))
        
        # Prefer shorter processing time (inverse)
        dur_score = 1.0 / (min_dur + 1e-9)
        
        # Combine: prioritize jobs with more remaining work, then shorter ops
        # Weight: 0.5 * remaining work + 0.3 * remaining ops + 0.2 * (1/duration)
        scores[i] = 0.5 * rw_score + 0.3 * ro_score + 0.2 * dur_score
        
        # Select the best machine: shortest duration
        best_machine = machines[int(np.argmin(durations))]
        # Store best machine in the op for potential use (if returning tuple)
        op["_best_machine"] = best_machine
    
    best_idx = int(np.argmax(scores))
    
    # Return just the index (the interface says "optionally (index, machine)")
    # Let's return just the index to be safe, as the seed does.
    return best_idx
# EVOLVE-BLOCK-END
