"""Combined Job-Shop Scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import math
import numpy as np

_START_TIME = _time.time()
_DEADLINE_SECONDS = 115.0  # Safety margin before 120s limit


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in a job-shop simulation.
    
    Strategy: Multi-criteria dispatch rule combining:
    1. Shortest Processing Time (SPT): Favor operations with shorter min duration.
    2. Critical Path / Remaining Work: Prioritize jobs with more remaining work.
    3. Machine Load Balance: Prefer machines that are less utilized.
    4. Remaining Operations: Favor jobs closer to completion to free up resources.
    
    We use a weighted composite score and pick the operation with the highest score.
    """
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE_SECONDS:
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each operation, find the minimum duration among machine options
    min_durations = np.zeros(n, dtype=np.float64)
    # Also track the best machine for each operation (for potential tie-breaking or load balance)
    best_machines = [None] * n
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            min_dur = float('inf')
            best_m = None
            for m, d in mo.items():
                if d < min_dur:
                    min_dur = d
                    best_m = m
            min_durations[i] = min_dur if min_dur < float('inf') else 1.0
            best_machines[i] = best_m
        else:
            min_durations[i] = 1.0

    # Normalize features to [0, 1]
    def normalize(arr):
        max_val = arr.max()
        min_val = arr.min()
        range_val = max_val - min_val
        if range_val < 1e-9:
            return np.full_like(arr, 0.5)
        return (arr - min_val) / range_val

    # 1. Shortest Processing Time: Inverse of min_duration (shorter = higher score)
    # Use a soft inverse to avoid extreme values
    inv_dur = 1.0 / (min_durations + 1.0)
    norm_spt = normalize(inv_dur)

    # 2. Remaining Work: Higher remaining work = higher priority (critical path)
    norm_rw = normalize(remaining_works)

    # 3. Remaining Ops: Fewer remaining ops = closer to completion = higher priority
    # Invert so that fewer ops get higher score
    inv_ops = 1.0 / (remaining_ops + 1.0)
    norm_ops = normalize(inv_ops)

    # 4. Machine Load Balance: We don't have direct machine load info, 
    #    but we can infer from the number of operations that prefer each machine.
    #    For now, use a simple heuristic: if multiple ops share the same best machine,
    #    slightly penalize to spread load.
    machine_counts = {}
    for m in best_machines:
        if m is not None:
            machine_counts[m] = machine_counts.get(m, 0) + 1
    
    load_penalty = np.zeros(n, dtype=np.float64)
    for i, m in enumerate(best_machines):
        if m is not None:
            load_penalty[i] = machine_counts[m]
    
    # Normalize load penalty (higher count = higher penalty)
    norm_load = normalize(load_penalty)
    # Invert so that less busy machines get higher score
    norm_load_inv = 1.0 - norm_load

    # Composite score with tuned weights
    # SPT is typically strong in job-shop scheduling
    # Remaining work captures critical path urgency
    # Remaining ops captures completion proximity
    # Load balance is a secondary tie-breaker
    w_spt = 0.35
    w_rw = 0.30
    w_ops = 0.20
    w_load = 0.15

    scores = w_spt * norm_spt + w_rw * norm_rw + w_ops * norm_ops + w_load * norm_load_inv

    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
