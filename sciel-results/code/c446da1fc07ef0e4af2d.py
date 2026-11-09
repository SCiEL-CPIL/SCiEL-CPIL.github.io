"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard
_START_TIME = None
_DEADLINE = 150.0

def _fast_dispatch(ready_operations, time):
    """Extremely fast fallback: Longest Remaining Work (LRW) / MWR."""
    if not ready_operations:
        return 0
    max_rw = -1
    best_idx = 0
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        if rw > max_rw:
            max_rw = rw
            best_idx = i
    return best_idx

def dispatch(ready_operations, time):
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 10.0:
        return _fast_dispatch(ready_operations, time)
    
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # 1. Calculate normalization constants
    max_rw = 0
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        if rw > max_rw:
            max_rw = rw
    if max_rw <= 0:
        max_rw = 1.0

    # 2. Estimate machine loads
    # Heuristic: Assume each op will be assigned to its minimum duration machine.
    # This gives a proxy for which machines are about to be congested.
    machine_load = {}
    total_min_duration = 0
    
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if not mo:
            continue
        min_d = min(mo.values())
        # Find the machine(s) with min duration
        # If multiple, we can spread the load or just pick one. 
        # Spreading is safer for load estimation.
        min_machines = [m for m, d in mo.items() if d == min_d]
        share = min_d / len(min_machines)
        for m in min_machines:
            machine_load[m] = machine_load.get(m, 0) + share
        total_min_duration += min_d
        
    if not machine_load:
        # Fallback if no machine options provided (shouldn't happen in valid FJSP)
        return _fast_dispatch(ready_operations, time)

    avg_load = sum(machine_load.values()) / len(machine_load)
    if avg_load <= 0:
        avg_load = 1.0

    # Weights
    w_rw = 0.60
    w_spt = 0.20
    w_load = 0.20
    
    best_score = -float('inf')
    best_idx = 0

    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        
        if not mo:
            # If no options, treat as high priority if high RW?
            # Or skip? The interface implies options exist.
            min_d = 1.0
            best_load_factor = 1.0
        else:
            durations = list(mo.values())
            min_d = min(durations)
            
            # Calculate effective cost for each machine
            # Cost = Duration * (1 + Load / AvgLoad)
            # We want to minimize this cost.
            min_effective_cost = float('inf')
            for m, d in mo.items():
                load = machine_load.get(m, 0)
                # Normalize load to be around 1.0 if average
                load_factor = load / avg_load
                cost = d * (1.0 + load_factor)
                if cost < min_effective_cost:
                    min_effective_cost = cost
            
            # SPT component: shorter is better
            # Use 1/(min_d + 1) to keep values bounded
            norm_spt = 1.0 / (min_d + 1.0)
            
            # Load component: lower effective cost is better
            # Normalize effective cost. 
            # If min_effective_cost is small, score is high.
            # Use 1 / (1 + min_effective_cost / avg_load) ?
            # Let's use a soft inverse.
            # If cost is 0, score 1. If cost is high, score 0.
            # Scale cost by avg_load to make it dimensionless.
            scaled_cost = min_effective_cost / avg_load
            norm_load = 1.0 / (1.0 + scaled_cost)

        # RW component
        norm_rw = rw / max_rw
        
        # Combined Score
        # We want to maximize the score.
        # High RW is good.
        # High SPT (short time) is good.
        # High Load Score (low cost) is good.
        
        score = (w_rw * norm_rw + 
                 w_spt * norm_spt + 
                 w_load * norm_load)
        
        if score > best_score:
            best_score = score
            best_idx = i
        elif abs(score - best_score) < 1e-9:
            # Tie breaker: 
            # 1. Prefer shorter min_d
            # 2. Prefer lower load
            prev_op = ready_operations[best_idx]
            prev_mo = prev_op.get("machine_options", {})
            prev_min_d = min(prev_mo.values()) if prev_mo else 1.0
            
            cur_min_d = min(mo.values()) if mo else 1.0
            
            if cur_min_d < prev_min_d:
                best_idx = i
            elif abs(cur_min_d - prev_min_d) < 1e-9:
                # Prefer lower remaining work? No, usually higher RW is better for makespan.
                # But if scores are tied, maybe break by Job ID for stability?
                # Or by remaining ops?
                # Let's stick to the main score logic.
                pass

    return best_idx

# EVOLVE-BLOCK-END
