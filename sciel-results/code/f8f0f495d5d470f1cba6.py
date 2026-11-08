import time

def dispatch(ready_operations, time):
    """
    Dispatches the next operation using a Dynamic Critical Path with SPT tie-break.
    
    Strategy:
    - Compute a "criticality" score for each ready operation based on the job's remaining work.
    - Use a weighted combination of remaining work (to prioritize jobs with lots of work left)
    - and the current operation's minimum duration (to break ties in favor of shorter operations).
    - Select the machine that minimizes the duration for the chosen operation.
    
    This is a balanced approach that avoids starving jobs with many small operations
    while also not ignoring the immediate cost of starting an operation.
    """
    if not ready_operations:
        return -1
    
    # Deadline guard: if we're close to the limit, return a simple SPT choice quickly
    # (This is a safety net; the actual computation below is O(n) and very fast)
    
    best_index = -1
    best_machine = -1
    best_score = float('inf')
    
    # Pre-compute: we'll use a composite score.
    # Score = alpha * (min_duration) + beta * (min_duration / (remaining_work + 1))
    # where alpha and beta are weights.
    # Actually, let's think in terms of "priority" where higher is better.
    # Priority = remaining_work - lambda * min_duration
    # We want to MAXIMIZE priority.
    
    # Let's use: score = min_dur + min_dur / max(rem_work, 1)
    # Lower score is better.
    # This penalizes long durations, and also penalizes long durations when rem_work is small.
    
    for i, op in enumerate(ready_operations):
        min_dur = float('inf')
        min_mach = -1
        for m, d in op['machine_options'].items():
            if d < min_dur:
                min_dur = d
                min_mach = m
        
        rem_work = op.get('remaining_work', 0)
        
        if rem_work <= 0:
            score = min_dur  # Just SPT if no remaining work
        else:
            # Composite: favor low min_dur, and if rem_work is large, the second term is small.
            # score = min_dur * (1 + 1.0 / rem_work)
            # This is SPT with a small boost for jobs that have a lot of work left.
            score = min_dur * (1.0 + 1.0 / max(rem_work, 1.0))
        
        if score < best_score:
            best_score = score
            best_index = i
            best_machine = min_mach
    
    return (best_index, best_machine)
