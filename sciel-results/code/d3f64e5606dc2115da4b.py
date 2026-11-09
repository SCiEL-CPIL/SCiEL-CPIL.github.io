"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math
import random

# Deadline guard
_START_TIME = None
_DEADLINE = 140.0  # Safety margin before 150s kill

def _soft_min(values, temperature=1.0):
    """Compute soft-min using log-sum-exp trick for stability."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    
    min_val = min(values)
    
    y_values = [(min_val - v) / temperature for v in values]
    
    max_y = max(y_values)
    if max_y > 500:
        return min_val
    
    sum_exp = sum(math.exp(y - max_y) for y in y_values)
    log_sum_exp = math.log(sum_exp) + max_y
    
    soft_min_val = -temperature * log_sum_exp + min_val
    
    if soft_min_val < min_val:
        soft_min_val = min_val
    
    return soft_min_val

def _estimate_job_completion(op, current_time):
    """
    Estimate when this job will complete if we start the current op now.
    Uses soft-min for each remaining operation's processing time.
    """
    remaining_ops = op.get("remaining_ops", 0)
    rw = op.get("remaining_work", 0)
    mo = op.get("machine_options", {})
    
    if not mo:
        return current_time + rw
    
    # Estimate average processing time per remaining op using soft-min
    durations = list(mo.values())
    sm_d = _soft_min(durations, temperature=0.3)
    
    # If we have remaining_ops info, use it to estimate total remaining time
    if remaining_ops > 0:
        # Current op takes sm_d, remaining ops take roughly sm_d each on average
        # But remaining_work gives us a better total estimate
        # Use a blend: assume remaining_work is spread over remaining_ops
        avg_per_op = rw / remaining_ops if remaining_ops > 0 else sm_d
        total_est = sm_d + (remaining_ops - 1) * avg_per_op
    else:
        total_est = rw
    
    return current_time + total_est

def dispatch(ready_operations, time):
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE:
        if not ready_operations:
            return 0
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    if n_ops == 1:
        return 0
    
    # Extract features
    features = []
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)
        
        if mo:
            durations = list(mo.values())
            sm_d = _soft_min(durations, temperature=0.3)
            min_d = min(durations)
            max_d = max(durations)
            n_machines = len(mo)
        else:
            sm_d = 1.0
            min_d = 1.0
            max_d = 1.0
            n_machines = 1
        
        # Job-level completion estimate
        job_completion = _estimate_job_completion(op, time)
        
        # Urgency ratio: remaining work per unit of effective processing time
        urgency_ratio = rw / sm_d if sm_d > 0 else rw
        
        features.append({
            'rw': rw,
            'sm_d': sm_d,
            'min_d': min_d,
            'max_d': max_d,
            'n_machines': n_machines,
            'remaining_ops': remaining_ops,
            'job_completion': job_completion,
            'urgency_ratio': urgency_ratio
        })
    
    # Normalize features
    max_rw = max(f['rw'] for f in features) or 1.0
    min_sm_d = min(f['sm_d'] for f in features)
    max_sm_d = max(f['sm_d'] for f in features) or 1.0
    min_n_machines = min(f['n_machines'] for f in features) or 1
    max_n_machines = max(f['n_machines'] for f in features) or 1
    max_completion = max(f['job_completion'] for f in features) or 1.0
    max_urgency = max(f['urgency_ratio'] for f in features) or 1.0
    
    # Weights tuned for job-level urgency and efficiency
    alpha = 0.40   # Weight for remaining work (urgency)
    beta = 0.25    # Weight for efficiency (shorter processing)
    gamma = 0.10   # Weight for flexibility
    delta_w = 0.15 # Weight for job completion time (earlier is better)
    epsilon = 0.10 # Weight for urgency ratio
    
    best_idx = 0
    best_score = -float('inf')
    
    for idx in range(n_ops):
        f = features[idx]
        
        rw_score = f['rw'] / max_rw
        
        if max_sm_d > min_sm_d:
            eff_score = 1.0 - (f['sm_d'] - min_sm_d) / (max_sm_d - min_sm_d)
        else:
            eff_score = 1.0
        
        if max_n_machines > min_n_machines:
            flex_score = (f['n_machines'] - min_n_machines) / (max_n_machines - min_n_machines)
        else:
            flex_score = 1.0
        
        # Job completion: lower is better (earlier completion)
        completion_score = 1.0 - (f['job_completion'] - time) / max_completion if max_completion > time else 1.0
        completion_score = max(0.0, min(1.0, completion_score))
        
        urgency_score = f['urgency_ratio'] / max_urgency
        
        score = alpha * rw_score + beta * eff_score + gamma * flex_score + delta_w * completion_score + epsilon * urgency_score
        
        if score > best_score:
            best_score = score
            best_idx = idx
        elif abs(score - best_score) < 1e-9:
            # Tie-breaker: prefer shorter soft-min duration, then more remaining ops
            if f['sm_d'] < features[best_idx]['sm_d']:
                best_idx = idx
            elif abs(f['sm_d'] - features[best_idx]['sm_d']) < 1e-9:
                if f['remaining_ops'] > features[best_idx]['remaining_ops']:
                    best_idx = idx
    
    return int(best_idx)

# EVOLVE-BLOCK-END
