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
    
    best_idx = 0
    best_score = -float('inf')
    
    for idx in range(n_ops):
        op = ready_operations[idx]
        mo = op.get("machine_options", {})
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
        else:
            min_d = 1.0
            
        # Calculate Longest Remaining Path (LRP)
        # LRP is the sum of the minimum possible durations of all remaining operations
        # This approximates the critical path length for the job
        remaining_ops = op.get("remaining_ops", [])
        
        if remaining_ops:
            lrp = 0.0
            for rem_op in remaining_ops:
                rem_mo = rem_op.get("machine_options", {})
                if rem_mo:
                    lrp += min(rem_mo.values())
                else:
                    lrp += 1.0 # Default duration if no options
        else:
            # If no remaining ops listed, assume just the current op
            lrp = min_d
            
        # Ensure LRP is at least the current min duration
        if lrp < min_d:
            lrp = min_d

        # Score: Prioritize jobs with high Critical Path (LRP)
        # Among those, prioritize operations with short processing time (MinDur)
        # This reduces the time the job occupies the system and allows subsequent
        # critical operations to start sooner.
        
        # Add a small constant to denominator to avoid division by zero and smooth
        # the difference between very short and short operations.
        score = lrp / (min_d + 0.1)
        
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
