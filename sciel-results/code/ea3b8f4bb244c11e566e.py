"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Explicit deadline guard: 150s wall-clock limit, use 145s for safety
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0


def _deadline_exceeded():
    return _time.time() >= _DEADLINE


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling using a refined Urgency/Cost ratio.
    
    Strategy:
    - Urgency: combines remaining_work with a smooth critical-path estimate
    - Cost: uses soft-min over machine durations (log-sum-exp style) for smoothness
    - Machine selection: balances current duration with future workload impact
    
    The ratio urgency/cost adapts dynamically as the schedule evolves.
    """
    if _deadline_exceeded():
        return 0
    
    if not ready_operations:
        return 0
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    for idx, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
        
        durations = list(machine_options.values())
        machines = list(machine_options.keys())
        
        if not durations:
            continue
        
        min_duration = min(durations)
        max_duration = max(durations)
        
        # Smooth critical-path estimate: use soft-max of durations scaled by remaining ops
        # This approximates the longest path through remaining operations
        if remaining_ops > 1:
            # Estimate critical path: current op duration + average remaining work per op
            # Use soft-max to smoothly weight between min and max durations
            temp = 0.1
            log_sum_exp = math.log(sum(math.exp(d * temp) for d in durations)) / temp
            soft_max_duration = log_sum_exp - math.log(len(durations))
            critical_path = soft_max_duration + (remaining_work - min_duration) / max(1, remaining_ops - 1)
        else:
            critical_path = min_duration
        
        # Urgency: remaining_work weighted with critical-path estimate
        # Jobs with longer critical paths are more urgent
        urgency = remaining_work + 0.3 * critical_path + 0.2 * max(0, 10 - remaining_ops)
        
        # Cost: soft-min over durations for smoothness (log-sum-exp with negative sign)
        if len(durations) > 1:
            temp = 0.1
            log_sum_exp_neg = math.log(sum(math.exp(-d * temp) for d in durations)) / temp
            soft_min_duration = -log_sum_exp_neg - math.log(len(durations))
        else:
            soft_min_duration = durations[0]
        
        cost = soft_min_duration + 1e-9
        
        # Score: urgency / cost (higher is better)
        score = urgency / cost
        
        if score > best_score:
            best_score = score
            best_idx = idx
            # Choose machine: balance shortest duration with load considerations
            # Prefer machines with shorter durations, but if durations are similar,
            # pick the one that leaves more capacity for future ops
            best_machine_score = -float('inf')
            best_machine_choice = machines[0]
            
            for m, d in zip(machines, durations):
                # Primary: shorter duration is better
                # Secondary: if durations are close, prefer machines that can handle
                # the remaining work more efficiently
                machine_score = -d
                # Small bonus if this machine's duration is well below average
                avg_d = sum(durations) / len(durations)
                if d < avg_d:
                    machine_score += 0.1 * (avg_d - d)
                
                if machine_score > best_machine_score:
                    best_machine_score = machine_score
                    best_machine_choice = m
            
            best_machine = best_machine_choice
    
    if best_machine is not None:
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
