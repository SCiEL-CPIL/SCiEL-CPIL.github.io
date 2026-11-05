import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return -1

    # Approach: Shortest Processing Time (SPT)
    # We select the operation that can be completed in the shortest time.
    # This keeps machines busy with quick tasks, potentially allowing
    # other operations to start sooner.
    
    def get_min_duration(op):
        options = op['machine_options']
        if not options:
            return float('inf')
        return min(options.values())

    # Find the operation with the minimum processing time
    # Tie-breaker: Job with maximum remaining work (to clear critical jobs)
    # Tie-breaker 2: Job with minimum remaining operations (to finish jobs)
    
    best_idx = -1
    best_score = None
    
    for i, op in enumerate(ready_operations):
        min_dur = get_min_duration(op)
        rem_work = op['remaining_work']
        rem_ops = op['remaining_ops']
        
        # Score tuple: (min_duration, -remaining_work, remaining_ops)
        # We want min min_duration, max remaining_work (so -rem_work), min rem_ops
        score = (min_dur, -rem_work, rem_ops)
        
        if best_score is None or score < best_score:
            best_score = score
            best_idx = i
            
    return best_idx
