"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Deadline guard: 140s is a safe margin before the 150s kill limit.
_START_TIME = _time.time()
_DEADLINE = 140.0


def dispatch(ready_operations, time):
    """
    Dispatch rule: Ratio-based priority combining processing time and remaining work.
    
    Strategy:
    1. For each ready operation, compute the minimum processing time across all
       machine options (shortest processing time, SPT).
    2. Use a ratio-based score that prioritizes operations with high remaining work
       relative to their processing time (critical path awareness).
    3. Add a small bonus for operations with fewer remaining steps to break ties.
    
    The ratio remaining_work / (min_duration + epsilon) emphasizes operations that
    have substantial remaining work but can be completed quickly, reducing idle time.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Check deadline
    if _time.time() > _DEADLINE:
        # Fallback to simple rule
        return int(max(range(n), key=lambda i: ready_operations[i]["remaining_work"]))
    
    scores = []
    for op in ready_operations:
        # Minimum processing time across machine options
        durations = list(op["machine_options"].values())
        min_duration = min(durations) if durations else 0
        
        # Remaining work and remaining ops
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Ratio-based score: high remaining work relative to processing time
        # Use epsilon to avoid division by zero
        epsilon = 1e-9
        ratio_score = remaining_work / (min_duration + epsilon)
        
        # Small bonus for operations closer to completion (fewer remaining ops)
        # This helps break ties and prioritize finishing jobs
        completion_bonus = 1.0 / (remaining_ops + 1) if remaining_ops > 0 else 0
        
        # Combined score
        score = ratio_score + 0.1 * completion_bonus
        
        scores.append(score)
    
    # Return index of operation with highest score
    return int(max(range(n), key=lambda i: scores[i]))
# EVOLVE-BLOCK-END
