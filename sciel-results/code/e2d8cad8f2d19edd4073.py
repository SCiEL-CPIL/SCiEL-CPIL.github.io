"""Flexible Job-Shop Scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import math
import numpy as np

# Deadline guard: evaluator kills at 150s; keep a safety margin.
_DEADLINE = _time.monotonic() + 140.0  # 10s safety margin


def _soft_min(values, temperature=0.1):
    """Compute a soft-min of a list of values using log-sum-exp trick."""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    min_val = np.min(arr)
    if temperature <= 1e-10:
        return float(min_val)
    shifted = arr - min_val
    weights = np.exp(-shifted / temperature)
    return float(min_val + temperature * np.log(np.sum(weights)))


def _soft_max(values, temperature=0.1):
    """Compute a soft-max of a list of values using log-sum-exp trick."""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    max_val = np.max(arr)
    if temperature <= 1e-10:
        return float(max_val)
    shifted = arr - max_val
    weights = np.exp(shifted / temperature)
    return float(max_val + temperature * np.log(np.sum(weights)))


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and optionally which machine.
    
    Args:
        ready_operations: list of dicts, each with:
            - job: job id
            - op: operation id
            - machine_options: dict {machine: duration}
            - remaining_ops: number of remaining operations in the job
            - remaining_work: total remaining work for the job
        time: current simulation time
        
    Returns:
        index of the ready operation to start, or (index, machine) tuple.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Deadline check
    if _time.monotonic() > _DEADLINE:
        return 0
    
    # For each ready operation, compute a priority score.
    # We want to minimize makespan, so we prioritize operations that:
    # 1. Have the most remaining work (to finish critical jobs early)
    # 2. Have few remaining operations (to unblock jobs)
    # 3. Can be done quickly on some machine (to keep machines busy)
    
    # Compute per-operation features
    remaining_works = []
    remaining_ops_list = []
    min_durations = []
    machine_counts = []
    
    for op in ready_operations:
        remaining_works.append(op["remaining_work"])
        remaining_ops_list.append(op["remaining_ops"])
        durations = list(op["machine_options"].values())
        min_durations.append(min(durations) if durations else 0)
        machine_counts.append(len(durations))
    
    # Normalize features to [0, 1] range for consistent weighting
    rw_max = max(remaining_works) if remaining_works else 1
    rw_min = min(remaining_works) if remaining_works else 0
    ro_max = max(remaining_ops_list) if remaining_ops_list else 1
    ro_min = min(remaining_ops_list) if remaining_ops_list else 0
    md_max = max(min_durations) if min_durations else 1
    md_min = min(min_durations) if min_durations else 0
    
    # Use soft-max/soft-min for smooth surrogate of the objective
    # We want to maximize a composite score that balances:
    # - High remaining work (process critical jobs)
    # - Low remaining ops (unblock jobs)  
    # - Low min duration (keep throughput high)
    
    temperature = 0.5
    
    # Compute normalized scores
    scores = []
    for i in range(n):
        # Normalize remaining work (higher is better for priority)
        rw_norm = (remaining_works[i] - rw_min) / (rw_max - rw_min + 1e-10)
        
        # Normalize remaining ops (lower is better for priority - fewer ops means closer to completion)
        ro_norm = 1.0 - (remaining_ops_list[i] - ro_min) / (ro_max - ro_min + 1e-10)
        
        # Normalize min duration (lower is better)
        md_norm = 1.0 - (min_durations[i] - md_min) / (md_max - md_min + 1e-10)
        
        # Composite score with weighted sum
        # Weights: emphasize remaining work most, then remaining ops, then duration
        w_rw = 0.5
        w_ro = 0.3
        w_md = 0.2
        
        score = w_rw * rw_norm + w_ro * ro_norm + w_md * md_norm
        scores.append(score)
    
    # Use soft-max to get a smooth selection (but we need a hard choice)
    # For the final decision, pick the argmax of scores
    best_idx = int(np.argmax(scores))
    
    # Now choose the best machine for the selected operation
    selected_op = ready_operations[best_idx]
    machine_options = selected_op["machine_options"]
    
    if machine_options:
        # Choose the machine with the shortest duration
        best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
        return (best_idx, best_machine)
    
    return best_idx
# EVOLVE-BLOCK-END
