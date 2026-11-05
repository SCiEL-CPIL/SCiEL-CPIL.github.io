"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 130.0  # safety margin before 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to schedule next.
    
    Strategy: 
    1. Prioritize jobs with Longest Remaining Processing Time (LRPT) to minimize critical path.
    2. Among those, prefer operations with Shortest Processing Time (SPT) on the assigned machine to reduce immediate machine occupation.
    3. Explicitly select the machine with the minimum duration for the chosen operation.
    """
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        return 0
    
    n = len(ready_operations)
    
    # Precompute features for each ready operation
    remaining_work = []
    remaining_ops = []
    min_duration = []
    best_machines = []
    
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            # Find the machine with the shortest duration
            best_machine = min(mo, key=mo.get)
            min_d = mo[best_machine]
        else:
            min_d = 0
            best_machine = None
        
        remaining_work.append(rw)
        remaining_ops.append(ro)
        min_duration.append(min_d)
        best_machines.append(best_machine)
    
    # Normalize features to [0, 1] range
    def normalize(vals):
        mn = min(vals)
        mx = max(vals)
        rng = mx - mn
        if rng < 1e-9:
            return [0.0] * len(vals)
        return [(v - mn) / rng for v in vals]
    
    norm_rw = normalize(remaining_work)
    norm_ro = normalize(remaining_ops)
    norm_min_d = normalize(min_duration)
    
    # Weights:
    # LRPT (remaining_work) is the strongest predictor of makespan reduction.
    # SPT (min_duration) helps keep machines free.
    w_rw = 0.6
    w_ro = 0.1
    w_min_d = 0.3
    
    # If many operations are ready, SPT becomes more important to maintain flow
    if n >= 6:
        w_min_d = 0.4
        w_rw = 0.5

    # Compute scores
    scores = []
    for i in range(n):
        s_rw = norm_rw[i]
        s_ro = norm_ro[i]
        s_min_d = 1.0 - norm_min_d[i]  # Shorter duration is better
        
        score = (w_rw * s_rw + 
                 w_ro * s_ro + 
                 w_min_d * s_min_d)
        scores.append(score)
    
    # Find best index
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score + 1e-9:
            best_score = scores[i]
            best_idx = i
    
    # Return index and the optimal machine choice
    return (best_idx, best_machines[best_idx])

# EVOLVE-BLOCK-END
