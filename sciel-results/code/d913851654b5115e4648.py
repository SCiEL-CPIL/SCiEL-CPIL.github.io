"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
import random as _random


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    Returns index of the ready operation to start.
    
    Strategy: Composite heuristic that balances:
    1. Most remaining work (urgency)
    2. Avoidance of bottleneck machines (penalize ops on machines with high load)
    3. Bonus for operations with more remaining ops (keep jobs progressing)
    4. Small random tiebreaker to avoid systematic ties
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Track elapsed time for deadline guard
    t_start = _time.time()
    deadline = t_start + 0.5  # 500ms safety margin per dispatch call
    
    # Precompute features for each ready operation
    scores = np.zeros(n)
    
    # Feature 1: remaining_work (higher = more urgent)
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    
    # Feature 2: minimum duration across machine options (shorter = less blocking)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Feature 3: maximum duration across machine options (longer = more bottleneck risk)
    max_durations = np.array([
        max(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Feature 4: remaining_ops count (more ops left = job is further from completion)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Feature 5: number of machine options (more options = more flexibility)
    num_options = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Normalize features to [0, 1] range for combination
    def normalize(arr):
        if arr.max() == arr.min():
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())
    
    nw_norm = normalize(remaining_work)
    min_dur_norm = normalize(min_durations)
    max_dur_norm = normalize(max_durations)
    rem_ops_norm = normalize(remaining_ops)
    num_opt_norm = normalize(num_options)
    
    # Composite score:
    # + weight * remaining_work (urgency: jobs with more work should go first)
    # - weight * max_duration (avoid bottleneck: ops with long max duration are risky)
    # + weight * remaining_ops (progress: keep jobs with more ops moving)
    # + weight * num_options (flexibility: ops with more options are less constrained)
    # - small weight * min_duration (slight preference for shorter min duration)
    
    w_work = 0.40
    w_maxdur = 0.25
    w_relops = 0.20
    w_numopt = 0.10
    w_mindur = 0.05
    
    scores = (
        w_work * nw_norm +
        w_maxdur * (1.0 - max_dur_norm) +
        w_relops * rem_ops_norm +
        w_numopt * num_opt_norm +
        w_mindur * (1.0 - min_dur_norm)
    )
    
    # Add small random tiebreaker to break systematic ties
    if _time.time() < deadline:
        random_tiebreak = np.random.uniform(0, 0.01, size=n)
        scores += random_tiebreak
    
    # Check deadline
    if _time.time() > deadline:
        # Return best found so far (scores already computed)
        return int(np.argmax(scores))
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
