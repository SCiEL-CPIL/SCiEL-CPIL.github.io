"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Deadline guard: 140s is a safe margin before the 150s kill limit.
_START_TIME = _time.time()
_DEADLINE = 140.0


def dispatch(ready_operations, time):
    """
    Dispatch rule: Weighted combination of processing time and remaining work.
    
    Strategy:
    1. For each ready operation, compute the minimum processing time across all
       machine options (shortest processing time, SPT).
    2. Combine this with the remaining work to create a priority score.
    3. Select the operation with the highest priority score.
    
    The combination weights short operations higher (to reduce blocking) while
    also considering the total remaining work (to prioritize critical jobs).
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
        
        # Average remaining work per operation
        avg_remaining = remaining_work / remaining_ops if remaining_ops > 0 else 0
        
        # Combined score: prioritize short operations and high remaining work
        # Weight: 0.5 * normalized min_duration + 0.5 * normalized remaining_work
        # We want to maximize remaining_work but minimize min_duration
        # So: score = remaining_work - alpha * min_duration
        # Let's use a weighted combination
        score = remaining_work - 0.3 * min_duration
        
        scores.append(score)
    
    # Return index of operation with highest score
    return int(max(range(n), key=lambda i: scores[i]))
# EVOLVE-BLOCK-END
