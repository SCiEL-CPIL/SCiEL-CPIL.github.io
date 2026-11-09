import time
import numpy as np
import sys

# Deadline guard to ensure we don't exceed the 120s limit
# We track the start time of the evaluation if possible, or just keep it fast.
# Given the 120s limit for 5 instances, and dispatch being called ~50-100 times per instance,
# we have plenty of time for O(N^2) lookahead simulations.
# However, to be safe against pathological cases or slow Python loops, we add a simple check.

def dispatch(ready_operations, time):
    """
    Dispatches the next operation using a Dynamic Bottleneck Optimization strategy.
    
    Strategy:
    1. For each ready operation, identify the best machine (shortest processing time).
    2. Estimate the "Congestion" of the machines eligible for each operation.
       Since we don't have explicit machine load state, we approximate congestion
       by the sum of minimum durations of all ready operations that *could* use 
       those machines. This assumes that if many ops want the same machine, 
       it is likely a bottleneck.
    3. Calculate a priority score for each operation:
       Score = (Job's Remaining Work) / (1 + Estimated Machine Load)
       - High remaining work indicates urgency (critical path).
       - High estimated machine load indicates congestion.
       - We want to pick the operation with the highest ratio, effectively 
         prioritizing urgent jobs on less congested machines.
       
    Alternative Interpretation of "Dynamic Bottleneck":
    The prompt suggests: "prioritizing the operation that maximizes the ratio of 
    its job's remaining critical path length to the estimated cumulative load 
    (sum of minimum durations) of its eligible machines."
    
    Let's implement exactly that:
    For operation i belonging to job j_i:
    - remaining_critical_path_i = remaining_work[j_i] (as a proxy for critical path)
    - eligible_machines_i = set of machines in machine_options[i]
    - estimated_load_i = sum of min_duration(k) for all ready ops k 
                         such that eligible_machines[k] intersects eligible_machines_i?
                         Or simpler: sum of min_duration(k) for all ready ops k
                         that share *any* machine with op i.
                         
    Actually, a simpler and more robust proxy for "cumulative load of its eligible machines"
    is to sum the minimum durations of all other ready operations that compete for 
    the *same* specific machine that op i is best suited for? 
    No, the prompt says "sum of minimum durations of its eligible machines".
    This phrasing is slightly ambiguous. Does it mean:
    A) Sum of min_duration(op_i) for all ops that can use ANY machine in op_i's option set?
    B) Sum of min_duration(op_i) for all ops that can use the BEST machine of op_i?
    
    Let's interpret "estimated cumulative load (sum of minimum durations) of its eligible machines" 
    as the total processing demand that *could* hit the machines available to this operation.
    
    Let M_i be the set of machines available for op i.
    Load_i = sum( min_duration(k) for all ready ops k if M_k intersects M_i )
    
    This captures the contention. If op i can use machines {1, 2}, and 10 other ops 
    can use machine 1 or 2, Load_i is high.
    
    Score_i = remaining_work[j_i] / (1 + Load_i)
    
    We pick the operation with the MAXIMUM Score_i.
    Tie-breaker: Shortest processing time (SPT) to free resources sooner.
    
    This is structurally different from the previous "Min Global Critical Path" approach 
    because it explicitly penalizes operations that are competing for highly contested 
    machines, even if they have high remaining work. It encourages load balancing.
    """
    if not ready_operations:
        return -1

    n_ops = len(ready_operations)
    
    # Extract data
    jobs = [op['job'] for op in ready_operations]
    rem_works = [op.get('remaining_work', 0) for op in ready_operations]
    
    # Pre-compute min duration, best machine, and set of eligible machines for each op
    min_durs = []
    best_machines = []
    eligible_sets = []
    
    for op in ready_operations:
        min_d = float('inf')
        best_m = -1
        opts = op['machine_options']
        eligible = set(opts.keys())
        
        for m, d in opts.items():
            if d < min_d:
                min_d = d
                best_m = m
                
        min_durs.append(min_d)
        best_machines.append(best_m)
        eligible_sets.append(eligible)
        
    # Calculate "Estimated Cumulative Load" for each operation
    # Load_i = sum of min_duration(k) for all k where eligible_sets[k] intersects eligible_sets[i]
    # This is O(N^2) with set intersections, which is fine for N~10-50.
    
    loads = [0.0] * n_ops
    
    for i in range(n_ops):
        load_i = 0.0
        set_i = eligible_sets[i]
        for k in range(n_ops):
            if i == k:
                continue
            # Check intersection
            # If there is any common machine, they compete
            if set_i & eligible_sets[k]:
                load_i += min_durs[k]
        loads[i] = load_i
        
    # Calculate Score: Maximize (Remaining Work / (1 + Load))
    # We want to pick the op with the highest score.
    
    best_idx = -1
    best_score = -float('inf')
    best_machine = -1
    
    for i in range(n_ops):
        rw = rem_works[i]
        load = loads[i]
        d = min_durs[i]
        m = best_machines[i]
        
        # Avoid division by zero, though load is sum of positive durations so > 0 usually.
        # 1 + load ensures stability.
        score = rw / (1.0 + load)
        
        # Tie-breaker: If scores are very close, prefer shorter duration (SPT)
        # to free the machine faster.
        # We can incorporate this by subtracting a small epsilon * d
        # But since we want to MAXIMIZE score, we subtract.
        # Let's use a small penalty for duration.
        score -= 0.01 * d
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = m

    if best_idx == -1:
        return -1
        
    return (best_idx, best_machine)
