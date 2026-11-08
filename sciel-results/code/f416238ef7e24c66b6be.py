"""Rewritten dispatch rule for job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 130.0  # safety margin before 150s limit


def _is_deadline_exceeded():
    return _time.time() > _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Uses a WSPT-based (Weighted Shortest Processing Time) heuristic:
    - Primary: remaining_work / min_duration (urgency ratio)
    - Secondary: inverse min_duration (prefer shorter ops)
    - Tertiary: inverse remaining_ops (prefer jobs closer to completion)
    
    Returns the index of the chosen operation.
    """
    if not ready_operations:
        return 0
    
    if _is_deadline_exceeded():
        # Fallback: pick the operation with minimum processing time
        min_idx = 0
        min_dur = float('inf')
        for i, op in enumerate(ready_operations):
            min_op_dur = min(op["machine_options"].values())
            if min_op_dur < min_dur:
                min_dur = min_op_dur
                min_idx = i
        return int(min_idx)
    
    n = len(ready_operations)
    scores = []
    
    for i, op in enumerate(ready_operations):
        # Get the minimum processing time across available machines
        min_dur = min(op["machine_options"].values())
        
        # Remaining work for the job
        rem_work = op["remaining_work"]
        
        # Number of remaining operations
        rem_ops = op["remaining_ops"]
        
        # WSPT-like ratio: remaining work / current op duration
        # Higher ratio means more work queued behind this op -> prioritize
        eps = 1e-9
        wspt_ratio = rem_work / (min_dur + eps)
        
        # Combine:
        # 1. WSPT ratio (primary): prioritize jobs with more pending work per unit time
        # 2. Inverse min_dur: prefer shorter ops to reduce blocking
        # 3. Inverse rem_ops: prefer jobs closer to completion
        score = 1.0 * wspt_ratio + 0.3 / (min_dur + eps) + 0.2 / (rem_ops + eps)
        
        scores.append(score)
    
    # Pick the operation with the highest score
    max_score_idx = 0
    max_score = scores[0]
    for i in range(1, n):
        if scores[i] > max_score:
            max_score = scores[i]
            max_score_idx = i
    
    return int(max_score_idx)
# EVOLVE-BLOCK-END
