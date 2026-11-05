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
    1. Critical Ratio (CR) - prioritize operations with tightest slack relative to work
    2. Longest Remaining Processing Time (LRPT) - prioritize jobs with more work
    3. Shortest Processing Time on assigned machine (SPT) - minimize immediate delay
    4. Most operations remaining - prioritize jobs with longer chains
    
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
    norm_min_d = normalize(min_duration)
    norm_avg_d = normalize(avg_duration)
    
    # Compute Critical Ratio (CR): slack / remaining_work
    # Slack is estimated as a proxy for time until job completion
    # Lower CR = more urgent = higher priority
    cr_values = []
    for i in range(n):
        rw = remaining_work[i]
        # Estimate slack: use remaining_work as proxy for time until completion
        # CR = slack / processing_time. Here we approximate slack with remaining_work
        # and processing_time with min_duration (or avg_duration)
        if rw > 0:
            # Use avg_duration as the processing time estimate
            proc_time = avg_duration[i] if avg_duration[i] > 0 else 1.0
            cr = rw / proc_time
        else:
            cr = float('inf')
        cr_values.append(cr)
    
    # Normalize CR: lower CR is better, so we negate for scoring
    norm_cr = normalize(cr_values)
    s_cr = [1.0 - v for v in norm_cr]  # Higher score for lower CR
    
    # Adaptive weights: 
    # - Higher weight on CR when there's significant variation
    # - Balance between urgency (CR) and work volume (LRPT)
    
    # Base weights
    w_cr = 0.35
    w_rw = 0.35
    w_ro = 0.10
    w_min_d = 0.15
    w_avg_d = 0.05
    
    # If few ready ops, be more aggressive with urgency
    if n <= 3:
        w_cr = 0.45
        w_rw = 0.35
        w_ro = 0.10
        w_min_d = 0.10
        w_avg_d = 0.00
    elif n >= 8:
        w_cr = 0.30
        w_rw = 0.30
        w_ro = 0.15
        w_min_d = 0.20
        w_avg_d = 0.05
    
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
        # Shorter duration = higher priority (negate since normalize gives 0=shortest)
        s_min_d = 1.0 - norm_min_d[i]
        s_avg_d = 1.0 - norm_avg_d[i]
        # Critical Ratio: lower CR = higher priority
        s_cr_i = s_cr[i]
        
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
        
        score = (w_cr * s_cr_i + 
                 w_rw * s_rw + 
                 w_ro * s_ro + 
                 w_min_d * s_min_d + 
                 w_avg_d * s_avg_d +
                 0.05 * s_contention)
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
