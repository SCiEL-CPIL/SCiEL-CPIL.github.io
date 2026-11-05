"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0

def _deadline_exceeded():
    return _time.time() >= _DEADLINE

def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Strategy:
    - Priority: Maximize (remaining_work / min_duration + 0.1 * remaining_ops)
      This balances Critical Path (remaining_work) with SPT (min_duration)
      and adds a slight preference for jobs with more remaining operations
      to keep long chains moving.
    - Machine Selection: Select the machine with the minimum processing time (SPT).
    """
    if _deadline_exceeded():
        return 0
    
    if not ready_operations:
        return 0
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    for idx, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
        
        durations = list(machine_options.values())
        machines = list(machine_options.keys())
        
        if not durations:
            continue
        
        min_duration = min(durations)
        
        if min_duration <= 0:
            min_duration = 1e-9
            
        # Core score: Urgency / Cost
        score = remaining_work / min_duration
        
        # Tie-breaker / Secondary objective: Prefer more remaining ops
        # This helps keep long critical paths active
        score += 0.1 * remaining_ops
        
        if score > best_score:
            best_score = score
            best_idx = idx
            
            # Find the machine with the minimum duration
            # If there are ties, pick the first one (stable)
            best_machine = min(machine_options.items(), key=lambda x: x[1])[0]
            
    if best_machine is not None:
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
