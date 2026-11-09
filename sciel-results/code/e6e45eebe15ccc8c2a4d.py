"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard
_START_TIME = None
_DEADLINE = 140.0  # Safety margin before 150s kill

def dispatch(ready_operations, time):
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: return first operation or a simple heuristic
        if not ready_operations:
            return 0
        # Simple fallback: Longest Remaining Work
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # If only one operation, return it
    if n_ops == 1:
        return 0
    
    # Compute features for each ready operation
    # Key insight: For flexible job-shop, we should prioritize operations that
    # are critical to the job completion (long remaining work) AND can be processed
    # quickly (short min duration on available machines).
    
    best_idx = 0
    best_score = -float('inf')
    
    for idx in range(n_ops):
        op = ready_operations[idx]
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
            n_machines = len(mo)
        else:
            min_d = 1.0
            n_machines = 1
        
        # Score: balance between urgency (remaining work) and efficiency (min duration)
        # Use a ratio-based approach: prefer ops with high rw/min_d ratio
        # This naturally handles normalization without needing global stats
        
        # Weighted score:
        # - rw gives urgency: jobs with more remaining work should be prioritized
        # - min_d gives efficiency: shorter processing times reduce makespan
        # - n_machines gives flexibility: more options means less congestion risk
        
        # Combine: score = rw / (min_d * (1 + 0.1 * (n_machines - 1)))
        # The denominator penalizes long processing times slightly more when there are
        # fewer machine options (less flexibility to avoid congestion)
        
        if min_d > 0:
            penalty = 1.0 + 0.1 * (n_machines - 1)
            score = rw / (min_d * penalty)
        else:
            score = rw
        
        if score > best_score:
            best_score = score
            best_idx = idx
        elif abs(score - best_score) < 1e-9:
            # Tie-breaker: prefer shorter min duration
            if mo:
                cur_min_d = min(mo.values())
                best_mo = ready_operations[best_idx].get("machine_options", {})
                best_min_d = min(best_mo.values()) if best_mo else 1.0
                if cur_min_d < best_min_d:
                    best_idx = idx
    
    return int(best_idx)

# EVOLVE-BLOCK-END
