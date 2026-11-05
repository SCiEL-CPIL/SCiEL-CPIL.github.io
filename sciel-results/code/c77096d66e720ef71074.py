"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    1. Select the operation with the highest composite score based on:
       - Remaining Work (Urgency)
       - Minimum Duration (Efficiency/Blocking time)
       - Number of Machine Options (Constrainedness)
    2. Return (index, machine) where machine is the one with the minimum duration
       for the selected operation, to minimize the blocking time on that machine.
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
        # If only one op, pick its best machine
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            best_m = min(opts, key=opts.get)
            return (0, best_m)
        return 0
    
    # Build scoring vectors
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    min_durations = np.empty(n, dtype=np.float64)
    best_machines = [None] * n
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            min_durations[i] = 0.0
            best_machines[i] = None
            continue
        
        # Find machine with minimum duration
        best_m = min(opts, key=opts.get)
        best_machines[i] = best_m
        min_durations[i] = opts[best_m]
        
        # Count options
        num_options = len(opts)
        # We'll use this in scoring

    # Normalize features
    rw_max = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    md_max = np.max(min_durations) if np.max(min_durations) > 0 else 1.0
    
    # Score components:
    # 1. Remaining Work (higher is better) - Weight 0.5
    score_rw = (remaining_works / rw_max) * 0.5
    
    # 2. Min Duration (lower is better) - Weight 0.3
    # Use 1/(md+eps) to handle zeros and emphasize short durations
    score_md = (1.0 / (min_durations + 1e-6))
    md_max_score = np.max(score_md) if np.max(score_md) > 0 else 1.0
    score_md = (score_md / md_max_score) * 0.3
    
    # 3. Flexibility (fewer options is better) - Weight 0.2
    # We need to re-calculate num_options or store it
    num_options = np.array([len(op["machine_options"]) for op in ready_operations], dtype=np.float64)
    # Invert: 1/num_options
    score_flex = (1.0 / (num_options + 1e-6))
    flex_max_score = np.max(score_flex) if np.max(score_flex) > 0 else 1.0
    score_flex = (score_flex / flex_max_score) * 0.2
    
    total_score = score_rw + score_md + score_flex
    
    # Small random tiebreaker
    rng = np.random.default_rng(seed=int(time * 1000) % (2**31))
    tiebreaker = rng.random(n) * 0.001
    total_score += tiebreaker
    
    best_idx = int(np.argmax(total_score))
    best_machine = best_machines[best_idx]
    
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx
# EVOLVE-BLOCK-END
