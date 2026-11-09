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
    
    # Clip temperature to avoid numerical issues
    temp = max(temperature, 1e-6)
    
    # Find min for numerical stability
    min_val = min(values)
    exp_vals = [math.exp(-(v - min_val) / temp) for v in values]
    sum_exp = sum(exp_vals)
    
    if sum_exp < 1e-30:
        return min_val
    
    return min_val + temp * math.log(sum_exp)

def dispatch(ready_operations, time):
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: return first operation or a simple heuristic
        if not ready_operations:
            return 0
        # Simple fallback: Longest Remaining Work
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # If only one operation, return it
    if n_ops == 1:
        return 0
    
    # For each ready operation, compute a comprehensive score
    # that estimates the impact on makespan.
    
    # Key insight: We want to minimize the maximum completion time across all jobs.
    # A good proxy is to prioritize operations that:
    # 1. Have high remaining work in their job (critical path)
    # 2. Can be completed quickly (short processing time on best machine)
    # 3. Have flexibility (multiple machine options) to avoid bottlenecks
    
    # Compute features for each operation
    scores = []
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            durations = list(mo.values())
            n_machines = len(mo)
            
            # Use soft-min to estimate effective processing time
            # This smoothly balances between the best and worst machines
            soft_min_d = _soft_min(durations, temperature=0.5)
            min_d = min(durations)
            avg_d = sum(durations) / len(durations)
        else:
            soft_min_d = 1.0
            min_d = 1.0
            avg_d = 1.0
            n_machines = 1
        
        # Job-level urgency: remaining work is a proxy for critical path length
        # Higher remaining work means this job is more "critical"
        
        # Machine flexibility: more options = less bottleneck risk
        flexibility = math.log(1 + n_machines)
        
        # Processing efficiency: shorter is better
        # Use soft_min_d for a smooth estimate
        efficiency = 1.0 / (1.0 + soft_min_d)
        
        # Combined score: we want to maximize this
        # Weight remaining work heavily (it's the dominant factor for makespan)
        # Add smaller weights for efficiency and flexibility
        score = 0.6 * rw + 0.25 * efficiency * 10 + 0.15 * flexibility * 5
        
        scores.append(score)
    
    # Select the operation with the highest score
    best_idx = max(range(n_ops), key=lambda i: scores[i])
    
    return int(best_idx)

# EVOLVE-BLOCK-END
