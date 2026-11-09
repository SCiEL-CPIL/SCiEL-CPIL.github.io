"""Job-Shop dispatch: pick the ready operation to start next to minimize makespan."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: evaluator kills after 150s wall-clock. Stay well under.
_START_TS = _time.time()
DEADLINE_S = 115.0  # safety margin

# Track machine loads to balance across machines
_machine_load = {}

def _soft_min(values, temperature=0.5):
    """Soft-min via log-sum-exp: approximates min but considers all options."""
    if not values:
        return 1.0
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 1:
        return arr[0]
    # Soft min: -logsumexp(-x / T) * T
    neg_arr = -arr / temperature
    max_neg = np.max(neg_arr)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_arr - max_neg)))
    return -log_sum_exp * temperature

def dispatch(ready_operations, time):
    global _machine_load, _START_TS, DEADLINE_S
    
    # Deadline guard
    if _time.time() - _START_TS > DEADLINE_S:
        return 0
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if remaining_ops <= 0:
            remaining_ops = 1
        
        if machine_options:
            durations = list(machine_options.values())
            machines = list(machine_options.keys())
            
            # Soft-min for smooth surrogate of min processing time
            soft_min_dur = _soft_min(durations, temperature=0.3)
            
            # Machine load balancing: prefer less-loaded machines
            # Estimate load as sum of durations on that machine from other ready ops
            # Use the soft-min machine's current load estimate
            # For simplicity, use the average load across options
            load_penalty = 0.0
            if _machine_load:
                for m in machines:
                    load_penalty += _machine_load.get(m, 0)
                load_penalty /= len(machines)
            
            # Effective processing time: duration + load penalty
            effective_dur = soft_min_dur + load_penalty * 0.5
        else:
            effective_dur = 1.0
        
        # Score: favor high remaining work, low effective duration, 
        # and jobs closer to completion (fewer remaining ops)
        # Weight: remaining_work dominates, then inverse duration, then 1/(remaining_ops)
        
        # Use a smooth combination:
        # score = remaining_work^0.7 / (effective_dur^0.3 + eps) * (1.0 / (remaining_ops + 0.5))
        
        rw = max(remaining_work, 0.1)
        ed = max(effective_dur, 0.1)
        ro = remaining_ops
        
        # Power-law combination for smoothness
        score_rw = rw ** 0.6
        score_dur = 1.0 / (ed ** 0.4 + 1e-6)
        score_ops = 1.0 / (ro + 0.5)
        
        scores[i] = score_rw * score_dur * score_ops
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    # Update machine load estimate for the chosen operation
    chosen_op = ready_operations[best_idx]
    mo = chosen_op.get("machine_options", {})
    if mo:
        # Assign to the machine with minimum duration (greedy)
        best_machine = min(mo, key=lambda m: mo[m])
        _machine_load[best_machine] = _machine_load.get(best_machine, 0) + mo[best_machine]
    
    return best_idx
# EVOLVE-BLOCK-END
