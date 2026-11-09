import numpy as np
import time as _time

_DEADLINE = None
_DEADLINE_MARGIN = 2.0  # seconds

def _ensure_deadline():
    """Set the global deadline once, based on first call time."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = _time.time() + 150.0 - _DEADLINE_MARGIN
    return _DEADLINE

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    deadline = _ensure_deadline()
    if _time.time() > deadline:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Extract metrics
    works = np.array([op["remaining_work"] for op in ready_operations])
    ops_counts = np.array([op["remaining_ops"] for op in ready_operations])
    
    # Calculate min duration and best machine for each op using soft-min
    # Soft-min gives a smooth approximation and also lets us consider machine balance
    min_durs = np.zeros(n)
    machine_loads = np.zeros(5)  # 5 machines max
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            min_durs[i] = 0.0
            continue
        durs = np.array(list(opts.values()))
        machines = np.array(list(opts.keys()))
        
        # Soft-min with temperature 0.5 for smoothness
        temp = 0.5
        log_durs = np.log(durs + 1e-10)
        max_log = np.max(log_durs)
        soft_min = max_log - np.log(np.sum(np.exp(log_durs - max_log)))
        min_durs[i] = np.exp(soft_min)
        
        # Track machine loads (approximate)
        for m, d in opts.items():
            if m < 5:
                machine_loads[m] += d * 0.1  # small weight for load estimation

    # Handle edge case where all works are 0
    if works.max() == 0:
        return 0

    # Adaptive Heuristic with smoother weighting
    work_spread = works.max() - works.min()
    avg_work = works.mean()
    
    # Compute alpha with a smooth sigmoid-like transition
    if avg_work > 0:
        ratio = work_spread / avg_work
        # Use a smooth function: alpha = 1.1 / (1 + exp(1.5 * (ratio - 1.25)))
        # This gives alpha ~ 1.0 when ratio is small, alpha ~ 0.1 when ratio is large
        alpha = 1.1 / (1.0 + np.exp(1.5 * (ratio - 1.25)))
        alpha = float(np.clip(alpha, 0.05, 1.0))
    else:
        alpha = 0.5
    
    # Add a small bonus for operations that can finish on less-loaded machines
    # For each op, find the machine with minimum (duration + machine_load_weight * load)
    machine_bonus = np.zeros(n)
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            continue
        best_score = np.inf
        for m, d in opts.items():
            m_idx = m if m < 5 else 0
            effective_cost = d + 0.2 * machine_loads[m_idx]
            if effective_cost < best_score:
                best_score = effective_cost
        if best_score < np.inf:
            machine_bonus[i] = min_durs[i] - best_score  # negative means better machine available

    # Score = Work - alpha * Duration + small machine balance bonus
    scores = works - alpha * min_durs + 0.1 * machine_bonus
    
    # Tie breaker: prefer operations with more remaining ops (completes jobs faster)
    # Only apply if scores are very close (within 1%)
    max_score = np.max(scores)
    if max_score > 0:
        close_threshold = max_score * 0.99
        close_mask = scores >= close_threshold
        if np.sum(close_mask) > 1:
            # Among close candidates, pick the one with most remaining ops
            close_indices = np.where(close_mask)[0]
            close_ops = ops_counts[close_indices]
            best_local = np.argmax(close_ops)
            best_idx = int(close_indices[best_local])
        else:
            best_idx = int(np.argmax(scores))
    else:
        best_idx = int(np.argmax(scores))
    
    return best_idx
