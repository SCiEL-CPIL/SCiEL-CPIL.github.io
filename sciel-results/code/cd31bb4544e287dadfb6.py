import numpy as np
import time as _time

_START_TIME = None
_DEADLINE = 115.0  # Safety margin before 150s kill

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    if not ready_operations:
        return 0
    
    # Deadline guard: if we are approaching the limit, fall back to a fast heuristic
    if _time.time() - _START_TIME > _DEADLINE:
        min_dur = float('inf')
        best_idx = 0
        for i, op in enumerate(ready_operations):
            min_op_dur = min(op["machine_options"].values())
            if min_op_dur < min_dur:
                min_dur = min_op_dur
                best_idx = i
        return int(best_idx)
    
    n_ops = len(ready_operations)
    if n_ops == 1:
        return 0
    
    try:
        # Machine-contention avoidance model:
        # Score each operation by how well it can be scheduled on under-loaded machines.
        # Key insight: an operation with a small minimum duration relative to its total
        # duration across all available machines has more flexibility to avoid contention.
        
        # First, compute current machine utilization from the ready queue
        machine_load = {}
        for op in ready_operations:
            for m, d in op["machine_options"].items():
                if m not in machine_load:
                    machine_load[m] = 0
                machine_load[m] += d
        
        # Normalize machine loads to [0, 1] range for consistent scaling
        if machine_load:
            max_load = max(machine_load.values())
            if max_load > 0:
                machine_load = {m: load / max_load for m, load in machine_load.items()}
        
        scores = []
        for op in ready_operations:
            durations = list(op["machine_options"].values())
            if not durations:
                scores.append(0.0)
                continue
            
            min_dur = min(durations)
            sum_dur = sum(durations)
            remaining_work = op["remaining_work"]
            remaining_ops = op["remaining_ops"]
            num_machines = len(op["machine_options"])
            
            # Handle zero duration
            if min_dur <= 0:
                min_dur = 1e-6
            if sum_dur <= 0:
                sum_dur = 1e-6
            
            # PRIMARY SIGNAL: Contention avoidance ratio
            # min_dur / sum_dur is high when the operation can be done quickly on one machine
            # relative to the total work across all machines. This means it has a "fast escape"
            # option that reduces its exposure to machine contention.
            contention_ratio = min_dur / sum_dur
            
            # SECONDARY SIGNAL: Minimum machine utilization among available machines
            # We want to pick operations whose best machine is currently least loaded.
            min_util = min(machine_load.get(m, 0.0) for m in op["machine_options"])
            
            # TERTIARY SIGNAL: Urgency based on remaining work
            # Jobs with more remaining work are more critical to the makespan
            urgency = remaining_work / (min_dur + 1e-6)
            
            # QUATERNARY SIGNAL: Flexibility bonus
            # Operations with more machine options have more ways to avoid contention
            flexibility_bonus = 0.1 * (num_machines - 1) if num_machines > 1 else 0.0
            
            # Composite score: prioritize operations that can be scheduled on under-loaded
            # machines and have good contention-avoidance characteristics
            score = (
                2.0 * contention_ratio +      # High ratio = good escape option
                1.5 * (1.0 - min_util) +      # Low utilization on best machine
                0.5 * urgency +               # More remaining work = more urgent
                flexibility_bonus             # More options = more flexibility
            )
            
            scores.append(score)
        
        return int(np.argmax(scores))
    except Exception:
        return 0
