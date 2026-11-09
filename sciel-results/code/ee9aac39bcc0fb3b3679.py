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
    v_min = np.min(v)
    # soft_min = -log(sum(exp(-v_i / T)) / n) * T
    # This gives a value close to min but slightly higher, reflecting the spread
    scaled = -v / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -temperature * (log_sum - log(len(v)))
    
    return float(soft_min_val)


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next using a multi-factor heuristic with soft-min surrogate.
    
    Factors considered:
    1. remaining_work: total work left for the job (higher = more urgent)
    2. remaining_ops: number of ops left (higher = more complex, start early)
    3. Machine flexibility: how many machines the op can run on (fewer = more constrained, prioritize)
    4. Soft-min duration: captures the full distribution of machine options, not just min
    5. Variance of durations: high variance indicates machine contention risk
    
    We use a weighted score combining these factors. The weight for
    remaining_work is dominant since it directly correlates with makespan.
    The soft-min provides a smoother signal than hard min, and variance
    captures the risk of machine contention.
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
    
    # Precompute factors
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        mo = op["machine_options"]
        
        # Machine flexibility: fewer options = more constrained
        num_machines = len(mo)
        flexibility = 1.0 / max(num_machines, 1)
        
        # Get all durations
        durations = list(mo.values()) if mo else [0]
        
        # Soft-min: smooth approximation that captures distribution
        # Use temperature to control how "soft" the min is
        # Lower temperature = closer to hard min; higher = more average-like
        soft_min_dur = _soft_min(durations, temperature=0.5)
        
        # Variance: high variance means some machines are much slower, indicating contention risk
        dur_arr = np.array(durations, dtype=np.float64)
        dur_var = float(np.var(dur_arr)) if len(dur_arr) > 1 else 0.0
        
        # Hard min for reference
        min_dur = float(np.min(dur_arr)) if len(dur_arr) > 0 else 0
        
        # Score components:
        # 1. Remaining work (dominant)
        # 2. Remaining ops (secondary urgency)
        # 3. Flexibility (constrained ops first)
        # 4. Soft-min duration (SPT effect, smooth version)
        # 5. Negative variance (prefer low variance = less contention risk)
        
        # We want to maximize urgency, so:
        # - Higher remaining_work -> higher score
        # - Higher remaining_ops -> higher score  
        # - Lower flexibility (fewer machines) -> higher score
        # - Lower soft_min_dur -> higher score (SPT)
        # - Lower variance -> higher score (less risk)
        
        scores[i] = (
            1.0 * rw +                          # dominant: most work remaining
            0.3 * ro +                          # secondary: more ops to complete
            2.0 * flexibility +                 # constrained ops get priority
            0.15 / (soft_min_dur + 1.0) +       # SPT: shorter ops preferred (smooth)
            -0.05 * dur_var                     # penalize high variance (contention risk)
        )
    
    # Select the operation with highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
