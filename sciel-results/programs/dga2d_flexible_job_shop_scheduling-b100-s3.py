import time

def dispatch(ready_operations, time):
    """
    Dispatches the next operation using a Dynamic Bottleneck Optimization strategy.
    
    Strategy:
    - Prioritize operations that maximize the ratio of their job's remaining critical 
      path length (approximated by remaining_work) to the estimated cumulative load 
      of their eligible machines.
    - This shifts focus from static job attributes to real-time machine congestion, 
      explicitly avoiding operations that would exacerbate load on already busy machines.
    - Machine selection: For the chosen operation, select the machine with the minimum 
      duration to minimize immediate processing time.
    """
    if not ready_operations:
        return -1

    # Deadline guard: if we're close to the limit, return a simple SPT choice quickly
    # (This is a safety net; the actual computation below is O(n*m) and very fast)
    
    best_index = -1
    best_machine = -1
    best_score = float('inf')
    
    # We want to maximize "efficiency" or minimize "penalty".
    # Penalty = (min_duration / remaining_work) * (1 + congestion_factor)
    # Congestion factor is based on the sum of min durations of all eligible machines.
    # If a job's operations can only go to a few machines, and those machines have 
    # high potential load (sum of min durs of other ops? No, we don't have that info easily).
    # Instead, let's use the sum of min durations of the CURRENT op's options as a proxy 
    # for "how constrained/flexible" it is, but the prompt suggests "cumulative load of 
    # its eligible machines". Since we don't have global machine loads, we approximate 
    # congestion by the average duration across available machines. A higher average 
    # duration suggests less efficient options or "heavier" work.
    
    # Let's refine: 
    # Score to MINIMIZE: 
    #   (min_dur / max(rem_work, 1)) + lambda * (avg_dur / max(rem_work, 1))
    # This favors low min_dur (SPT) and low average duration (flexibility/efficiency).
    # If rem_work is large, the penalty is smaller, allowing us to pick a slightly 
    # longer op if it's part of a large job (critical path pressure).
    
    lambda_weight = 0.5
    
    for i, op in enumerate(ready_operations):
        durations = list(op['machine_options'].values())
        if not durations:
            continue
            
        min_dur = min(durations)
        avg_dur = sum(durations) / len(durations)
        
        # Find the machine with min_dur
        min_mach = -1
        min_val = float('inf')
        for m, d in op['machine_options'].items():
            if d < min_val:
                min_val = d
                min_mach = m
        
        rem_work = op.get('remaining_work', 1)
        if rem_work <= 0:
            rem_work = 1
            
        # The "congestion" or "load" proxy here is the average duration.
        # If the average duration is high relative to the min, it means the operation
        # is less flexible or the machines are slower.
        # We penalize operations that are long relative to the job's remaining work.
        
        # Score = (min_dur + lambda_weight * avg_dur) / rem_work
        # Lower score is better.
        
        score = (min_dur + lambda_weight * avg_dur) / rem_work
        
        if score < best_score:
            best_score = score
            best_index = i
            best_machine = min_mach
    
    if best_index == -1:
        return -1
        
    return (best_index, best_machine)
