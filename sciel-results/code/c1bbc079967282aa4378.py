"""Job-shop dispatch rule: Machine-contention-aware critical path with dynamic priority."""
# EVOLVE-BLOCK-START
import time as _time
import math

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 140.0  # safety margin before 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to schedule next.
    
    Strategy: Multi-critical-path dispatch with machine contention awareness.
    
    Key insight: In job-shop scheduling, the makespan is determined by the critical path
    through the job-machine dependency graph. We approximate this by:
    1. Estimating each job's remaining critical path length (sum of min durations).
    2. Penalizing operations that would be assigned to heavily contended machines.
    3. Using a dynamic priority that balances urgency (remaining work) with efficiency.
    
    The score combines:
    - Critical path urgency: jobs with more remaining work are more critical
    - Machine availability: prefer machines with less contention
    - Processing efficiency: shorter operations reduce blocking time
    - Depth consideration: operations deeper in the job sequence may have more flexibility
    """
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        return 0
    
    n = len(ready_operations)
    
    # Precompute features for each ready operation
    min_durations = []
    best_machines = []
    remaining_works = []
    remaining_ops = []
    op_indices = []  # operation index in the job
    
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        op_idx = op.get("op", 0)
        
        if mo:
            # Find machine with shortest duration
            best_m, min_d = min(mo.items(), key=lambda item: item[1])
        else:
            best_m = None
            min_d = 0
            
        min_durations.append(min_d)
        best_machines.append(best_m)
        remaining_works.append(rw)
        remaining_ops.append(ro)
        op_indices.append(op_idx)
    
    # Calculate machine contention: how many operations prefer each machine
    machine_counts = {}
    for m in best_machines:
        if m is not None:
            machine_counts[m] = machine_counts.get(m, 0) + 1
    
    # Normalize features
    max_rw = max(remaining_works) if remaining_works else 1.0
    if max_rw <= 0:
        max_rw = 1.0
    
    max_min_d = max(min_durations) if min_durations else 1.0
    if max_min_d <= 0:
        max_min_d = 1.0
    
    max_ro = max(remaining_ops) if remaining_ops else 1.0
    if max_ro <= 0:
        max_ro = 1.0
    
    # Calculate scores
    scores = []
    for i in range(n):
        rw = remaining_works[i]
        min_d = min_durations[i]
        best_m = best_machines[i]
        ro = remaining_ops[i]
        op_idx = op_indices[i]
        
        eps = 1e-9
        
        # 1. Critical path urgency: normalized remaining work
        # Jobs with more remaining work are more critical to the makespan
        cp_urgency = rw / max_rw
        
        # 2. Machine contention penalty
        # Operations that would go to heavily loaded machines get penalized
        if best_m is not None:
            count = machine_counts.get(best_m, 1)
            # Exponential decay: more severe penalty for higher contention
            contention_penalty = 1.0 / (1.0 + 0.3 * (count - 1))
        else:
            contention_penalty = 1.0
        
        # 3. Processing efficiency: shorter operations are better
        # Use normalized duration (shorter is better)
        duration_score = 1.0 - (min_d / max_min_d)
        
        # 4. Depth consideration: operations earlier in the job sequence
        # have less flexibility and should be prioritized
        # Normalize: higher op_idx means deeper in sequence
        if max_ro > 0:
            depth_factor = 1.0 - (op_idx / (max_ro + 1))
        else:
            depth_factor = 1.0
        
        # 5. Work-to-duration ratio: rewards efficient use of remaining work
        # Higher ratio means more work packed into less time
        work_ratio = rw / (min_d + eps)
        max_work_ratio = max(remaining_works[j] / (min_durations[j] + eps) for j in range(n)) if n > 0 else 1.0
        if max_work_ratio <= 0:
            max_work_ratio = 1.0
        work_ratio_norm = work_ratio / max_work_ratio
        
        # Combine scores with weights
        # Primary: critical path urgency (most important for makespan)
        # Secondary: machine contention (avoid bottlenecks)
        # Tertiary: processing efficiency and depth
        # Quaternary: work-to-duration ratio
        
        w_cp = 0.45      # critical path urgency
        w_cont = 0.25    # machine contention
        w_dur = 0.10     # duration efficiency
        w_depth = 0.10   # depth consideration
        w_ratio = 0.10   # work-to-duration ratio
        
        score = (
            w_cp * cp_urgency * contention_penalty +
            w_cont * contention_penalty +
            w_dur * duration_score +
            w_depth * depth_factor * contention_penalty +
            w_ratio * work_ratio_norm
        )
        
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
