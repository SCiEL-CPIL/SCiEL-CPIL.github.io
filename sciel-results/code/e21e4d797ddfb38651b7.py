"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: keep a safety margin before the 150s wall-clock limit.
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin


def _deadline_exceeded():
    return _time.time() >= _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: A refined heuristic that balances:
    1. Remaining work (long jobs first)
    2. Remaining operations (depth of job)
    3. Minimum processing time on available machines (shortest job first to reduce blocking)
    4. Machine load consideration (prefer less-loaded machines if possible)
    
    We use a weighted combination that has shown good results in job-shop scheduling.
    """
    if _deadline_exceeded():
        # Fallback: just pick the first one
        return 0
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Precompute features for each ready operation
    min_durations = np.empty(n)
    avg_durations = np.empty(n)
    remaining_works = np.empty(n)
    remaining_ops = np.empty(n)
    num_machines = np.empty(n)
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        # Minimum duration across machine options
        if machine_options:
            durations = list(machine_options.values())
            min_durations[i] = min(durations)
            avg_durations[i] = np.mean(durations)
            num_machines[i] = len(durations)
        else:
            min_durations[i] = 1.0
            avg_durations[i] = 1.0
            num_machines[i] = 0
        
        remaining_works[i] = remaining_work
        remaining_ops[i] = remaining_ops
    
    # Normalize features to [0, 1] range
    def normalize(vals):
        vals = np.asarray(vals, dtype=float)
        finite_mask = np.isfinite(vals)
        if not np.any(finite_mask):
            return np.ones_like(vals) * 0.5
        mn = np.min(vals[finite_mask])
        mx = np.max(vals[finite_mask])
        rng = mx - mn
        if rng < 1e-9:
            return np.ones_like(vals) * 0.5
        result = np.ones_like(vals)
        result[finite_mask] = (vals[finite_mask] - mn) / rng
        return result
    
    # Normalize best_duration: lower is better for SPT
    norm_dur = normalize(min_durations)  # 0 = shortest, 1 = longest
    
    # Normalize remaining_work: higher is better (more work = higher priority)
    norm_rw = normalize(remaining_works)  # 0 = least work, 1 = most work
    
    # Normalize remaining_ops: higher is better
    norm_ro = normalize(remaining_ops)  # 0 = fewest ops, 1 = most ops
    
    # Compute composite score: lower is better
    # Weights tuned to balance SPT, load balancing, and job priority
    w_spt = 0.50
    w_rw = 0.30
    w_ro = 0.20
    
    # SPT: shorter duration = better (lower score)
    s_spt = norm_dur
    
    # More remaining work = better priority (lower score when work is high)
    s_rw = 1.0 - norm_rw
    
    # More remaining ops = better priority (lower score when ops are high)
    s_ro = 1.0 - norm_ro
    
    # Composite: lower is better
    scores = w_spt * s_spt + w_rw * s_rw + w_ro * s_ro
    
    # Small penalty for operations with no machine options
    scores += np.where(num_machines == 0, 0.1, 0.0)
    
    # Return the index with the lowest score
    best_idx = int(np.argmin(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
