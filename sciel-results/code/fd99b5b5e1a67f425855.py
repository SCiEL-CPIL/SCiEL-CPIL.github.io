"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from math import log, exp

# Global deadline guard: 150s wall-clock kill; keep a safety margin.
_DEADLINE = 150.0
_START_TIME = _time.time()
_MARGIN = 10.0  # seconds of safety margin


def _elapsed_ok():
    return (_time.time() - _START_TIME) < (_DEADLINE - _MARGIN)


def _soft_min(values, temperature=1.0):
    """
    Compute a soft minimum using log-sum-exp.
    Returns a value that smoothly approximates min(values) but captures the full distribution.
    """
    if not values:
        return 0.0
    v = np.array(values, dtype=np.float64)
    if len(v) == 1:
        return float(v[0])
    
    # Use log-sum-exp trick for numerical stability
    scaled = -v / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -temperature * (log_sum - log(len(v)))
    
    return float(soft_min_val)


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next using a multi-factor heuristic with soft-min surrogate.
    
    Factors considered:
    1. Work density: remaining_work / remaining_ops (urgency per op)
    2. Remaining work total (dominant makespan driver)
    3. Machine flexibility: fewer options = more constrained
    4. Soft-min duration: smooth SPT signal
    5. Machine contention: how many ready ops compete for each machine
    6. Variance of durations: contention risk indicator
    
    We use a weighted score. Work density and total remaining work are dominant.
    Machine contention penalty spreads load across machines.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Quick path for small n
    if n == 1:
        return 0
    
    if not _elapsed_ok():
        # Fallback to simple rule if we're running out of time
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Precompute machine contention: count how many ready ops list each machine
    machine_contention = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo:
            machine_contention[m] = machine_contention.get(m, 0) + 1
    
    # Precompute factors
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = max(op["remaining_ops"], 1)
        mo = op["machine_options"]
        
        # Work density: urgency per remaining operation
        work_density = rw / ro
        
        # Machine flexibility: fewer options = more constrained
        num_machines = len(mo) if mo else 1
        flexibility = 1.0 / max(num_machines, 1)
        
        # Get all durations
        durations = list(mo.values()) if mo else [0]
        dur_arr = np.array(durations, dtype=np.float64)
        
        # Soft-min with temperature=1.0: balanced between hard min and mean
        soft_min_dur = _soft_min(durations, temperature=1.0)
        
        # Variance: high variance means machine contention risk
        dur_var = float(np.var(dur_arr)) if len(dur_arr) > 1 else 0.0
        
        # Machine contention: average contention across this op's machine options
        if mo:
            contention_vals = [machine_contention.get(m, 0) for m in mo]
            avg_contention = sum(contention_vals) / len(contention_vals)
        else:
            avg_contention = 0
        
        # Score components (maximize):
        # 1. Work density (urgency per op) - dominant
        # 2. Total remaining work (makespan driver)
        # 3. Flexibility (constrained ops first)
        # 4. SPT via soft-min inverse
        # 5. Negative contention (spread load)
        # 6. Negative variance (less risk)
        
        scores[i] = (
            1.2 * work_density +                # dominant: urgency per remaining op
            0.6 * rw +                          # total work remaining
            1.5 * flexibility +                 # constrained ops get priority
            0.25 / (soft_min_dur + 1.0) +       # SPT: shorter ops preferred
            -0.3 * avg_contention +             # penalize contended machines
            -0.05 * dur_var                     # penalize high variance
        )
    
    # Select the operation with highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
