"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: the evaluator kills after 150s wall-clock.
# We track elapsed time and bail out to a safe fallback if we approach the limit.
_T0 = _time.perf_counter()
_DEADLINE = 140.0  # safety margin of 10s


def _elapsed_ok():
    return (_time.perf_counter() - _T0) < _DEADLINE


def _soft_min(values, beta=2.0):
    """
    Soft-min via log-sum-exp surrogate.
    For a vector v, soft_min(v) ≈ min(v) - (1/beta) * log(sum(exp(-beta*v_i / min_v)))
    This captures the full distribution of machine durations, not just the minimum.
    """
    if len(values) == 0:
        return 1.0
    v = np.array(values, dtype=np.float64)
    v_min = np.min(v)
    # Shift for numerical stability
    shifted = v - v_min
    # log-sum-exp of -beta * shifted
    beta_shifted = -beta * shifted
    max_beta = np.max(beta_shifted)
    lse = max_beta + np.log(np.sum(np.exp(beta_shifted - max_beta)))
    return v_min - lse / beta


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy: Use a composite heuristic that balances:
    1. Remaining work (more work = higher priority)
    2. Remaining operations (more ops = higher priority, as they constrain completion)
    3. Soft-min machine duration (captures full distribution of machine options)
    4. Machine flexibility (number of options)
    
    The soft-min surrogate provides a smooth, informative measure of the "effective"
    processing time considering all available machines, which better reflects
    machine contention and load balancing opportunities.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Quick deadline check
    if not _elapsed_ok():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        mo = op["machine_options"]
        
        # Soft-min duration across available machines
        if mo:
            durations = list(mo.values())
            soft_min_dur = _soft_min(durations, beta=2.0)
            min_dur = min(durations)
            max_dur = max(durations)
            # Spread: how much variance in machine options (higher = more contention risk)
            spread = max_dur - min_dur
            n_opts = len(mo)
        else:
            soft_min_dur = 1.0
            min_dur = 1.0
            max_dur = 1.0
            spread = 0.0
            n_opts = 1
        
        # Heuristic components:
        # 1. Remaining work: more work means more critical
        # 2. Remaining ops: more ops means this job is further from completion
        # 3. Soft-min duration: prefer shorter effective processing times
        # 4. Spread penalty: prefer operations with less variance in machine options
        # 5. Machine flexibility: more options = less likely to cause bottlenecks
        
        # Normalize-ish scaling (avoid division by zero)
        score = (
            rw * 1.0 +            # primary: remaining work
            ro * 0.5 +             # secondary: remaining operations
            - soft_min_dur * 0.4 + # prefer shorter effective ops
            - spread * 0.15 +      # prefer less variance (more predictable)
            n_opts * 0.1           # prefer more flexible ops
        )
        scores[i] = score
    
    # Pick the operation with highest score
    idx = int(np.argmax(scores))
    
    return idx
# EVOLVE-BLOCK-END
