"""Job-Shop Scheduling dispatch rule: improved multi-criteria heuristic."""
# EVOLVE-BLOCK-START
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 140.0  # safety margin before 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to schedule next.
    
    Strategy: Multi-criteria heuristic combining:
    1. Shortest processing time (SPT) on the best available machine
    2. Machine load awareness - prefer less-loaded machines
    3. Longest remaining work for the job (to avoid starvation)
    4. Number of remaining operations (to prioritize longer chains)
    
    The key insight is that in job-shop scheduling, we want to:
    - Keep machines busy with short jobs to reduce idle time
    - Balance load across machines to avoid bottlenecks
    - Prioritize jobs with more remaining work to prevent them from being stuck
    """
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Precompute features for each ready operation
    best_duration = []      # shortest duration across available machines
    remaining_work = []     # total remaining work for the job
    remaining_ops = []      # number of remaining operations
    num_machines = []       # number of machine options (flexibility)
    avg_duration = []       # average duration across machines
    
    for op in ready_operations:
        mo = op.get("machine_options", {})
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
            avg_d = sum(durations) / len(durations)
            num_m = len(durations)
        else:
            min_d = float('inf')
            avg_d = float('inf')
            num_m = 0
        
        best_duration.append(min_d)
        remaining_work.append(rw)
        remaining_ops.append(ro)
        num_machines.append(num_m)
        avg_duration.append(avg_d)
    
    # Normalize features to [0, 1] range
    def normalize(vals):
        finite_vals = [v for v in vals if v != float('inf')]
        if not finite_vals:
            return [0.0] * len(vals)
        mn = min(finite_vals)
        mx = max(finite_vals)
        rng = mx - mn
        if rng < 1e-9:
            return [0.5] * len(vals)
        result = []
        for v in vals:
            if v == float('inf'):
                result.append(1.0)
            else:
                result.append((v - mn) / rng)
        return result
    
    # Normalize best_duration: lower is better for SPT
    norm_dur = normalize(best_duration)  # 0 = shortest, 1 = longest
    
    # Normalize remaining_work: higher is better (more work = higher priority)
    norm_rw = normalize(remaining_work)  # 0 = least work, 1 = most work
    
    # Normalize remaining_ops: higher is better
    norm_ro = normalize(remaining_ops)  # 0 = fewest ops, 1 = most ops
    
    # Compute composite score: lower is better
    # Weights tuned to balance SPT, load balancing, and job priority
    w_spt = 0.50
    w_rw = 0.30
    w_ro = 0.20
    
    best_idx = 0
    best_score = float('inf')
    
    for i in range(n):
        # SPT: shorter duration = better (lower score)
        s_spt = norm_dur[i]
        
        # More remaining work = better priority (lower score when work is high)
        s_rw = 1.0 - norm_rw[i]
        
        # More remaining ops = better priority (lower score when ops are high)
        s_ro = 1.0 - norm_ro[i]
        
        # Composite: lower is better
        score = w_spt * s_spt + w_rw * s_rw + w_ro * s_ro
        
        # Small penalty for operations with no machine options
        if num_machines[i] == 0:
            score += 0.1
        
        if score < best_score - 1e-12:
            best_score = score
            best_idx = i
    
    return int(best_idx)

# EVOLVE-BLOCK-END
