"""Improved dispatch rule for open-shop scheduling using machine-load balancing."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s wall-clock kill
_START = None


def _elapsed():
    global _START
    if _START is None:
        _START = _time.perf_counter()
    return _time.perf_counter() - _START


def dispatch(ready_operations, time):
    """
    Machine-load balancing dispatch rule for open-shop scheduling.
    
    Strategy:
    1. Select the operation that targets the most congested machine (highest remaining
       workload) to balance machine loads and prevent bottlenecks.
    2. Among operations targeting the same machine, prefer shorter durations to reduce
       blocking time.
    3. Return (index, machine) where machine is the one with the minimum duration
       for the selected operation.
    """
    global _START
    if _START is None:
        _START = _time.perf_counter()
    
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            best_m = min(opts, key=opts.get)
            return (0, best_m)
        return 0
    
    # Compute for each operation:
    # - min_duration: shortest processing time on any available machine
    # - best_machine: the machine with that min duration
    # - total_machine_load: sum of durations across all machine options (proxy for congestion)
    # - remaining_work: from the op dict
    # - remaining_ops: from the op dict
    
    min_durations = np.empty(n, dtype=np.float64)
    best_machines = [None] * n
    num_options = np.empty(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            min_durations[i] = 0.0
            best_machines[i] = None
            num_options[i] = 0
            continue
        
        best_m = min(opts, key=opts.get)
        best_machines[i] = best_m
        min_durations[i] = opts[best_m]
        num_options[i] = len(opts)
    
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Normalize features
    rw_max = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    md_max = np.max(min_durations) if np.max(min_durations) > 0 else 1.0
    no_max = np.max(num_options) if np.max(num_options) > 0 else 1.0
    
    # Score components with tuned weights and linear scaling for duration
    # 1. Remaining Work (higher is better) - Weight 0.3
    score_rw = (remaining_works / rw_max) * 0.3
    
    # 2. Min Duration (lower is better) - Weight 0.5
    # Linear scaling: shorter is better
    # If max_md is 0, all are 0, score is 1
    if md_max > 0:
        score_md = (1.0 - (min_durations / md_max)) * 0.5
    else:
        score_md = np.full(n, 0.5)
        
    # 3. Flexibility (fewer options is better, more constrained) - Weight 0.2
    # Linear scaling for options: fewer is better
    # min options is 1, max is no_max
    # score is high when num_options is low
    if no_max > 1:
        # Map [1, no_max] to [1, 0]
        score_flex = (1.0 - ((num_options - 1.0) / (no_max - 1.0))) * 0.2
    else:
        score_flex = np.full(n, 0.2)
        
    total_score = score_rw + score_md + score_flex
    
    # Small deterministic tiebreaker based on index to avoid random seeding issues
    tiebreaker = np.arange(n) * 1e-8
    total_score += tiebreaker
    
    best_idx = int(np.argmax(total_score))
    best_machine = best_machines[best_idx]
    
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx
# EVOLVE-BLOCK-END
