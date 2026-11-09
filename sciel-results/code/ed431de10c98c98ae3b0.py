# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def _fast_schedule(durations, demands, capacities, preds, priority_scores):
    """
    Fast serial schedule generation scheme.
    Uses a simple list-based approach but optimized with pre-computed successors and in-degrees.
    """
    n = len(durations)
    n_res = len(capacities)
    
    # Pre-compute graph structure
    in_degree = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    remaining_in_degree = in_degree.copy()
    sched_start = np.zeros(n)
    sched_end = np.zeros(n)
    scheduled = [False] * n
    
    # Resource usage over time (using a dictionary or array if time is discrete/small)
    # Since we don't know the horizon, we use a list of events or a simple check.
    # For speed, we'll use a "active activities" list and check feasibility by scanning.
    # Optimization: Keep a list of scheduled activities sorted by start time? 
    # Or just maintain a resource profile if time is integer.
    # Let's assume durations are integers or can be treated as floats but we scan integer steps.
    
    current_time = 0.0
    scheduled_count = 0
    active = [] # List of (end_time, index)
    
    # We need to find the earliest feasible start time for a given activity.
    # This requires checking resource availability.
    
    while scheduled_count < n:
        # Find eligible activities
        eligible = [i for i in range(n) if not scheduled[i] and remaining_in_degree[i] == 0]
        if not eligible:
            break # Should not happen in valid DAG
            
        # Sort by priority (higher is better)
        eligible.sort(key=lambda i: priority_scores[i], reverse=True)
        best = eligible[0]
        
        # Calculate earliest start based on precedence
        es_prec = 0.0
        for p in preds[best]:
            if sched_end[p] > es_prec:
                es_prec = sched_end[p]
                
        # Find earliest feasible start based on resources
        # We scan forward from es_prec
        # To speed this up, we can jump to the next time a resource becomes available.
        # But for simplicity and robustness, we scan integer steps.
        # If durations are large, this is slow. But n=30 is small.
        
        # Remove finished activities from active list to save memory/checks
        # Actually, we just need to check resources at time t.
        
        t = int(np.ceil(es_prec))
        # Check if resources are available at t
        while True:
            feasible = True
            for j in range(n_res):
                if demands[best][j] == 0:
                    continue
                used = 0.0
                # Check all scheduled activities that overlap with [t, t+duration)
                # Optimization: Only check activities that are currently active or start soon
                # For n=30, checking all is O(30*4) = 120 ops. Very fast.
                for k in range(n):
                    if scheduled[k]:
                        if sched_start[k] < t and sched_end[k] > t:
                            used += demands[k][j]
                if used + demands[best][j] > capacities[j] + 1e-9:
                    feasible = False
                    break
            
            if feasible:
                break
            t += 1
            
        start_time = float(t)
        sched_start[best] = start_time
        sched_end[best] = start_time + durations[best]
        scheduled[best] = True
        scheduled_count += 1
        
        for v in successors[best]:
            remaining_in_degree[v] -= 1
            
    return float(np.max(sched_end))

def priority(instance):
    start_time = time.time()
    DEADLINE = 140.0 # Safety margin
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    preds = instance["predecessors"]
    n = len(durations)
    n_res = len(capacities)
    
    # 1. Compute Topological Order
    in_degree = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
            
    topo_order = []
    rem = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            rem[v] -= 1
            if rem[v] == 0:
                queue.append(v)
                
    if len(topo_order) != n:
        # Fallback if cycle (shouldn't happen)
        topo_order = list(range(n))
        
    reverse_topo = list(reversed(topo_order))
    
    # 2. Compute Critical Path Length (CPL) - Tail Time
    cpl = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            cpl[u] = durations[u]
        else:
            cpl[u] = durations[u] + max(cpl[v] for v in successors[u])
            
    # 3. Compute Earliest Start (ES)
    es = np.zeros(n, dtype=float)
    for u in topo_order:
        if not preds[u]:
            es[u] = 0.0
        else:
            es[u] = max(es[p] + durations[p] for p in preds[u])
            
    # 4. Compute Latest Start (LS) based on CPL
    # Project makespan without resources
    project_length = max(es[i] + durations[i] for i in range(n))
    ls = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            ls[u] = project_length - durations[u]
        else:
            ls[u] = min(ls[v] - durations[u] for v in successors[u]) # Wait, this is not quite right for LS definition
            # LS[u] = min(LS[v]) - durations[u] ? No.
            # LS[u] = min_{v in succ(u)} (LS[v]) - durations[u] is correct if LS[v] is the latest start of v.
            # Actually, standard definition:
            # LF[u] = min_{v in succ(u)} (LF[v])
            # LS[u] = LF[u] - durations[u]
            # Let's stick to Slack = LS - ES.
            
    # Recalculate LS properly
    lf = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            lf[u] = project_length
        else:
            lf[u] = min(lf[v] for v in successors[u])
        ls[u] = lf[u] - durations[u]
        
    slack = ls - es
    
    # 5. Compute Resource Intensity
    # Ratio of demand to capacity, normalized
    res_intensity = np.zeros(n, dtype=float)
    for i in range(n):
        total_ratio = 0.0
        for j in range(n_res):
            if capacities[j] > 0:
                total_ratio += demands[i][j] / capacities[j]
        res_intensity[i] = total_ratio
        
    # 6. Define Priority Scores
    # Strategy: MinSlack is a very strong heuristic.
    # Priority = -Slack + alpha * ResourceIntensity
    # We want to schedule low slack first.
    
    # Let's try a few candidate rules and pick the best one quickly.
    # Candidates:
    # 1. -Slack (MinSlack)
    # 2. -Slack + 0.1 * ResIntensity
    # 3. -Slack + 0.5 * ResIntensity
    # 4. -CPL (Longest Processing Time / Critical Path)
    # 5. -Slack * ResIntensity (Hybrid)
    
    candidates = []
    
    # Normalize features to [0,1] for consistent weighting
    def norm(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-9:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
        
    n_slack = norm(slack)
    n_res = norm(res_intensity)
    n_cpl = norm(cpl)
    
    # Candidate 1: MinSlack
    candidates.append(-n_slack)
    
    # Candidate 2: MinSlack + Resource Pressure
    candidates.append(-n_slack + 0.2 * n_res)
    candidates.append(-n_slack + 0.5 * n_res)
    candidates.append(-n_slack + 1.0 * n_res)
    
    # Candidate 3: Critical Path
    candidates.append(-n_cpl)
    
    # Candidate 4: Weighted Slack (Slack * Res)
    candidates.append(-n_slack * (1 + n_res))
    
    # Evaluate candidates
    best_score = None
    best_makespan = float('inf')
    
    # We have plenty of time. Let's evaluate all candidates.
    for c in candidates:
        try:
            m = _fast_schedule(durations, demands, capacities, preds, c)
            if m < best_makespan:
                best_makespan = m
                best_score = c
        except Exception:
            continue
            
    # If we have time, do a local search on the weights of MinSlack + Res
    if time.time() - start_time < DEADLINE - 10:
        # Local search around the best candidate if it was one of the linear combos
        # Or just try a grid of alpha for -Slack + alpha * Res
        base_slack = -n_slack
        base_res = n_res
        
        # Grid search alpha
        best_alpha = 0.0
        best_m = best_makespan
        
        # Coarse search
        for alpha in np.linspace(0, 2, 11):
            c = base_slack + alpha * base_res
            try:
                m = _fast_schedule(durations, demands, capacities, preds, c)
                if m < best_m:
                    best_m = m
                    best_alpha = alpha
                    best_score = c
            except Exception:
                continue
                
        # Fine search around best_alpha
        if best_alpha > 0:
            for alpha in np.linspace(max(0, best_alpha - 0.2), min(2, best_alpha + 0.2), 11):
                c = base_slack + alpha * base_res
                try:
                    m = _fast_schedule(durations, demands, capacities, preds, c)
                    if m < best_m:
                        best_m = m
                        best_alpha = alpha
                        best_score = c
                except Exception:
                    continue
                    
    if best_score is None:
        # Fallback to MinSlack
        best_score = -n_slack
        
    return best_score

# EVOLVE-BLOCK-END
