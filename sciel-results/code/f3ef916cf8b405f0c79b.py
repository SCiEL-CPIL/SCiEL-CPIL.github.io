import numpy as np
import time

def priority(instance):
    t_start = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0

    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (n, n_res)
    capacities = np.asarray(instance["capacities"], dtype=float)  # (n_res,)
    predecessors = instance["predecessors"]  # list of lists
    
    n = len(durations)
    n_res = len(capacities)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Forward pass: ES, EF
    es = np.zeros(n)
    ef = np.zeros(n)
    for i in range(n):
        for p in predecessors[i]:
            es[i] = max(es[i], ef[p])
        ef[i] = es[i] + durations[i]
    
    project_end = np.max(ef)
    
    # Backward pass: LS, LF
    ls = np.zeros(n)
    lf = np.zeros(n)
    for i in range(n-1, -1, -1):
        if not successors[i]:
            lf[i] = project_end
        else:
            lf[i] = min(ls[s] for s in successors[i])
        ls[i] = lf[i] - durations[i]
    
    # Total float
    total_float = ls - es
    
    # Tail: distance from EF to project end
    tail = project_end - ef
    
    # Resource demand intensity: total demand relative to capacity
    if n_res > 0:
        demand_ratios = demands / np.maximum(capacities, 1e-9)  # (n, n_res)
        max_demand_ratio = np.max(demand_ratios, axis=1)  # (n,)
        # Sum of demand ratios (total resource pressure)
        sum_demand_ratio = np.sum(demand_ratios, axis=1)  # (n,)
    else:
        max_demand_ratio = np.zeros(n)
        sum_demand_ratio = np.zeros(n)
    
    # Normalize features
    max_float = np.max(total_float) if np.max(total_float) > 0 else 1.0
    max_tail = np.max(tail) if np.max(tail) > 0 else 1.0
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    max_ratio = np.max(max_demand_ratio) if np.max(max_demand_ratio) > 0 else 1.0
    max_sum_ratio = np.max(sum_demand_ratio) if np.max(sum_demand_ratio) > 0 else 1.0
    
    # Check deadline before computing final score
    if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
        # Return a simple fallback based on criticality
        return (max_float - total_float) / max_float
    
    # Priority: higher is more urgent
    # Emphasize CPM signals (float and tail) which are strongest predictors
    # Add resource contention awareness: activities that consume more of the
    # scarcest resources should be scheduled earlier to avoid bottlenecks
    
    w_float = 0.45
    w_tail = 0.30
    w_dur = 0.10
    w_res_max = 0.08
    w_res_sum = 0.07
    
    priority_scores = (
        w_float * (max_float - total_float) / max_float +
        w_tail * tail / max_tail +
        w_dur * durations / max_dur +
        w_res_max * max_demand_ratio / max_ratio +
        w_res_sum * sum_demand_ratio / max_sum_ratio
    )
    
    # Ensure non-negative scores
    priority_scores = np.maximum(priority_scores, 0.0)
    
    return priority_scores
