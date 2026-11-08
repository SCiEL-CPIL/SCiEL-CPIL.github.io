"""Rewritten dispatch rule for job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s limit


def _is_deadline_exceeded():
    return _time.time() > _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Uses a multi-criteria heuristic combining:
    1. Minimum remaining processing time on the best available machine
    2. Weighted by remaining work (to prioritize jobs with more work)
    3. Fewer remaining operations (to complete jobs sooner)
    
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
        
        # Heuristic score:
        # - Prefer shorter processing times (reduces machine blocking)
        # - Prefer jobs with more remaining work (balances load)
        # - Prefer jobs with fewer remaining operations (completes jobs sooner)
        
        # Normalize components to be comparable
        # Use a weighted combination:
        # score = w1 * (1/min_dur) + w2 * (rem_work) + w3 * (1/rem_ops)
        
        # To avoid division by zero, add small epsilon
        eps = 1e-9
        score = (1.0 / (min_dur + eps)) + 0.1 * rem_work + 0.5 / (rem_ops + eps)
        
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
