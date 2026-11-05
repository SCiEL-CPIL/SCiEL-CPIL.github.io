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
    
    Strategy: Multi-criteria heuristic combining:
    1. Longest Remaining Processing Time (LRPT) - prioritize jobs with more work
    2. Shortest Processing Time on assigned machine (SPT) - minimize immediate delay
    3. Most operations remaining - prioritize jobs with longer chains
    4. Earliest due date proxy - prioritize jobs with tightest constraints
    
    We use a weighted combination that adapts based on the current situation.
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
    max_duration = []
    avg_duration = []
    
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
            max_d = max(durations)
            avg_d = sum(durations) / len(durations)
        else:
            min_d = 0
            max_d = 0
            avg_d = 0
        
        remaining_work.append(rw)
        remaining_ops.append(ro)
        min_duration.append(min_d)
        max_duration.append(max_d)
        avg_duration.append(avg_d)
    
    # Normalize features to [0, 1] range for fair weighting
    def normalize(vals):
        mn = min(vals)
        mx = max(vals)
        rng = mx - mn
        if rng < 1e-9:
            return [0.0] * len(vals)
        return [(v - mn) / rng for v in vals]
    
    norm_rw = normalize(remaining_work)
    norm_ro = normalize(remaining_ops)
    norm_avg_d = normalize(avg_duration)
    
    # Compute NSPT: normalized shortest processing time weighted by remaining work
    # Lower NSPT = shorter duration relative to remaining work = better
    nspt_values = []
    for i in range(n):
        # Use min_duration normalized by remaining_work to get NSPT
        # Add epsilon to avoid division by zero
        rw_safe = remaining_work[i] + 1.0
        nspt = min_duration[i] / rw_safe
        nspt_values.append(nspt)
    
    norm_nspt = normalize(nspt_values)
    # For NSPT, lower is better, so we negate for scoring
    
    # Adaptive weights:
    # - NSPT (Normalized Shortest Processing Time): scale min_duration by remaining_work
    #   to combine SPT's machine-freeing with MWRT's job-progress guarantee
    # - This avoids starvation of long jobs while preferring quick ops among comparable jobs
    
    # Base weights
    w_rw = 0.30
    w_ro = 0.10
    w_nspt = 0.50  # NSPT: min_duration / (remaining_work + 1)
    w_avg_d = 0.10
    
    # If few ready ops, be more aggressive with remaining work
    if n <= 3:
        w_rw = 0.45
        w_ro = 0.10
        w_nspt = 0.35
        w_avg_d = 0.10
    elif n >= 8:
        w_nspt = 0.45
        w_rw = 0.35
        w_ro = 0.10
        w_avg_d = 0.10
    
    # Compute machine contention: how many ready ops target each machine
    machine_counts = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo:
            machine_counts[m] = machine_counts.get(m, 0) + 1
    
    # Compute scores (higher is better for our selection)
    scores = []
    for i in range(n):
        # More remaining work = higher priority
        s_rw = norm_rw[i]
        # More remaining ops = higher priority  
        s_ro = norm_ro[i]
        # NSPT: lower normalized value = shorter duration relative to remaining work = better
        s_nspt = 1.0 - norm_nspt[i]
        s_avg_d = 1.0 - norm_avg_d[i]
        
        # Machine contention: prefer ops going to less-contended machines
        # Find the machine with minimum duration for this op
        mo = ready_operations[i].get("machine_options", {})
        if mo:
            best_machine = min(mo, key=lambda m: mo[m])
            contention = machine_counts.get(best_machine, 1)
            # Normalize contention: fewer ops targeting this machine is better
            max_contention = max(machine_counts.values()) if machine_counts else 1
            s_contention = 1.0 - (contention - 1) / max(1, max_contention - 1)
        else:
            s_contention = 0.5
        
        score = (w_rw * s_rw + 
                 w_ro * s_ro + 
                 w_nspt * s_nspt + 
                 w_avg_d * s_avg_d +
                 0.1 * s_contention)
        scores.append(score)
    
    # Add a small tiebreaker: prefer lower op index for stability
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score + 1e-9:
            best_score = scores[i]
            best_idx = i
    
    return int(best_idx)

# EVOLVE-BLOCK-END
