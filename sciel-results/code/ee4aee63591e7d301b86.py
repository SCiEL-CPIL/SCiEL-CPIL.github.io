"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Multi-criteria weighted scoring combining:
    - MWR (Most Work Remaining): 0.5 weight — critical path priority
    - SPT (Shortest Processing Time): 0.3 weight — frees machines sooner
    - Fewer remaining ops: 0.2 weight — completes jobs, reduces tail risk
    
    Plus small adjustments for machine constraint tightness.
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0
    
    n = len(ready_operations)
    
    # If only one operation, no choice
    if n == 1:
        return 0
    
    # Precompute for normalization
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    min_durations = np.array([min(op["machine_options"].values()) if op["machine_options"] else 1.0 for op in ready_operations])
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])
    num_options = np.array([len(op["machine_options"]) if op["machine_options"] else 1 for op in ready_operations])
    
    # Normalize each criterion to [0, 1] range for fair weighting
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    # MWR: higher remaining work = higher priority
    mwr_norm = normalize(remaining_works)
    
    # SPT: shorter min duration = higher priority (invert)
    spt_norm = 1.0 - normalize(min_durations)
    
    # Remaining ops: fewer = higher priority (invert)
    rem_ops_norm = 1.0 - normalize(remaining_ops)
    
    # Machine constraint: fewer options = slightly higher priority (invert)
    constraint_norm = 1.0 - normalize(num_options)
    
    # Weighted composite score
    # Weights: MWR 0.5, SPT 0.3, remaining_ops 0.15, constraint 0.05
    scores = (0.5 * mwr_norm +
              0.3 * spt_norm +
              0.15 * rem_ops_norm +
              0.05 * constraint_norm)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
