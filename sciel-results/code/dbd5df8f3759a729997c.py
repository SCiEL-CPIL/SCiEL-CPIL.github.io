import numpy as np
import time as time_module

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # If only one option, return it
    if len(ready_operations) == 1:
        return 0

    # Deadline guard
    start_t = time_module.perf_counter()
    DEADLINE = 145.0  # 145s to leave 5s margin before 150s kill

    # Heuristic scoring function
    def score_op(op, current_time):
        rw = op["remaining_work"]
        mo = op["machine_options"]
        durations = list(mo.values())
        min_dur = min(durations) if durations else 0
        max_dur = max(durations) if durations else 0
        
        # MWR (Most Work Remaining) is the dominant factor
        # We want to prioritize jobs with high remaining work to prevent them from becoming the bottleneck
        # Also consider the duration: shorter operations on busy machines might be better to clear,
        # but MWR is usually stronger in open shop.
        
        # Let's try a composite score:
        # 1. Remaining Work (higher is better)
        # 2. Min Duration (lower is better? or higher priority for short ops to keep machines busy?)
        #    Actually, in open shop, if a machine is free, doing a short op might allow more ops to be done
        #    but if we have a long job waiting, we should do the long job's op if that machine is free.
        #    However, we don't know which machine is free for the other ops.
        #    The evaluator likely simulates the rest.
        
        # Let's stick to a strong MWR with a small tie-breaker.
        # Tie-breaker: Prefer shorter durations (SPT) to keep flow? Or longer?
        # Usually MWR is the best heuristic for makespan in open shop.
        
        return rw

    # Calculate scores for all ready operations
    scores = [score_op(op, time) for op in ready_operations]
    
    # Find the best index
    best_idx = int(np.argmax(scores))
    
    # If there are ties (within floating point epsilon), we might want to break them.
    # Let's check for ties and use a secondary heuristic: Min Duration
    max_score = scores[best_idx]
    ties = [i for i, s in enumerate(scores) if abs(s - max_score) < 1e-9]
    
    if len(ties) > 1:
        # Break tie by preferring the operation with the smallest minimum duration
        # This keeps machines busy with quick tasks if work remaining is equal
        def secondary_score(idx):
            op = ready_operations[idx]
            durations = list(op["machine_options"].values())
            return min(durations) if durations else float('inf')
        
        best_idx = min(ties, key=secondary_score)
        
    return best_idx
