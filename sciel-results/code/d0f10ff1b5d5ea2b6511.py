"""Job-shop dispatch rule: WSPT-enhanced with machine contention smoothing."""
# EVOLVE-BLOCK-START
import time as _time
import math

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 135.0  # safety margin before 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to schedule next.
    
    Strategy: Weighted Shortest Processing Time (WSPT) with contention penalty.
    1. Calculate WSPT score: remaining_work / (min_duration + epsilon).
       - Prioritizes jobs with more remaining work (urgency).
       - Prioritizes operations with shorter processing times (efficiency).
    2. Apply contention penalty:
       - Count how many ready operations prefer each machine (based on min duration).
       - Penalize operations targeting heavily contended machines.
    3. Select the operation with the highest adjusted score.
    """
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        return 0
    
    n = len(ready_operations)
    
    # Precompute features
    min_durations = []
    best_machines = []
    remaining_works = []
    
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            # Find machine with shortest duration
            best_m, min_d = min(mo.items(), key=lambda item: item[1])
        else:
            best_m = None
            min_d = 0
            
        min_durations.append(min_d)
        best_machines.append(best_m)
        remaining_works.append(rw)
        
    # Calculate machine contention
    # A machine is "contended" if many operations have it as their shortest-duration option
    machine_counts = {}
    for m in best_machines:
        if m is not None:
            machine_counts[m] = machine_counts.get(m, 0) + 1
    
    # Calculate scores
    scores = []
    for i in range(n):
        rw = remaining_works[i]
        min_d = min_durations[i]
        best_m = best_machines[i]
        
        # WSPT: Weight / Processing Time
        # Use epsilon to avoid division by zero
        eps = 1e-6
        wspt_score = rw / (min_d + eps)
        
        # Contention penalty
        if best_m is not None:
            count = machine_counts.get(best_m, 1)
            # Soft penalty: 1 / (1 + count)
            # If count is 1 (only this op), factor is 0.5
            # If count is 5, factor is 1/6 ~ 0.16
            # This heavily penalizes contended machines
            contention_factor = 1.0 / (1.0 + count)
        else:
            contention_factor = 1.0
            
        # Final score
        score = wspt_score * contention_factor
        scores.append(score)
        
    # Select best index
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
            
    return int(best_idx)

# EVOLVE-BLOCK-END
