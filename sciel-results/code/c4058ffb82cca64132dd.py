# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Greedy Bottleneck Elimination with Optimal Machine Selection.
    
    Strategy:
    1. For each ready operation, determine the minimum possible processing time
       (by selecting the fastest machine). This represents the operation's 
       intrinsic 'cost' or bottleneck potential.
    2. Prioritize operations with the highest minimum processing time (LPT).
       This ensures long, inflexible tasks are started early.
    3. Break ties by maximizing the remaining work of the job. Jobs with more
       remaining work are more critical to the overall makespan and should be
       kept progressing.
    4. Return the index of the selected operation and the specific machine
       that minimizes its processing time.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Arrays to store metrics
    min_times = np.zeros(n)
    best_machines = [None] * n
    remaining_works = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            min_times[i] = 0
            best_machines[i] = None
        else:
            # Find the machine with the minimum duration
            best_machine, min_duration = min(options.items(), key=lambda item: item[1])
            min_times[i] = min_duration
            best_machines[i] = best_machine
        
        remaining_works[i] = op["remaining_work"]
    
    # Primary Criterion: Maximize Minimum Processing Time (LPT)
    max_min_time = np.max(min_times)
    
    # Find all candidates with the max min_time
    candidates = [i for i in range(n) if min_times[i] == max_min_time]
    
    if len(candidates) == 1:
        idx = int(candidates[0])
        return (idx, best_machines[idx])
    
    # Secondary Criterion: Maximize Remaining Work
    # Prioritize jobs that have more work left, as they are more likely to be on the critical path
    max_work_val = max(remaining_works[i] for i in candidates)
    candidates2 = [i for i in candidates if remaining_works[i] == max_work_val]
    
    if len(candidates2) == 1:
        idx = int(candidates2[0])
        return (idx, best_machines[idx])
    
    # Tertiary Criterion: Minimize Remaining Ops (if work is tied, finish shorter jobs? 
    # Or maybe just pick the first one? Let's stick to a deterministic tie-break)
    # Actually, if work is tied, maybe prefer the one with fewer ops to clear the board?
    min_ops_val = min(ready_operations[i]["remaining_ops"] for i in candidates2)
    candidates3 = [i for i in candidates2 if ready_operations[i]["remaining_ops"] == min_ops_val]
    
    idx = int(candidates3[0])
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
