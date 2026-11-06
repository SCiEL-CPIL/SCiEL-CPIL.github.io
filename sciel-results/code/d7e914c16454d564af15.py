"""Combined RCPSP heuristic: static critical-path scoring + simulation-based bottleneck detection."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time


def priority(instance):
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    if n == 0:
        return np.array([], dtype=float)

    # Get predecessors with defensive key handling
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred", "precedence"]:
        if key in instance:
            preds = instance[key]
            break

    if preds is None:
        return np.asarray(durations, dtype=float)

    # Normalize preds to list of lists of ints
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()

    # Build successor list
    succs = [set() for _ in range(n)]
    for j in range(n):
        if j < len(preds):
            p_list = preds[j]
            if p_list is None:
                continue
            for i in p_list:
                if i is not None:
                    i = int(i)
                    if 0 <= i < n and i != j:
                        succs[i].add(j)

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return np.asarray(durations, dtype=float)

    # Topological sort
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in succs[j]:
            in_degree[i] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)

    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)

    # Forward pass: ES
    ES = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in succs[u]:
            if ES[u] + durations[u] > ES[v]:
                ES[v] = ES[u] + durations[u]

    project_duration = np.max(ES + durations) if n > 0 else 0

    # Backward pass: LS
    LS = np.full(n, project_duration, dtype=float)
    for u in reversed(topo_order):
        for v in succs[u]:
            if LS[v] - durations[u] < LS[u]:
                LS[u] = LS[v] - durations[u]

    total_float = LS - ES

    # Resource data
    demands = None
    capacities = None
    for key in ["demands", "resource_demands", "demand", "resource_demand"]:
        if key in instance:
            demands = instance[key]
            break
    for key in ["capacities", "resource_capacities", "capacity", "resource_capacity"]:
        if key in instance:
            capacities = instance[key]
            break

    demands_arr = None
    capacities_arr = None
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        if demands_arr.ndim == 1:
            demands_arr = demands_arr.reshape(-1, 1)
        if capacities_arr.ndim == 0:
            capacities_arr = capacities_arr.reshape(1)
        if demands_arr.shape[0] == n and capacities_arr.shape[0] > 0:
            safe_cap = np.where(capacities_arr > 0, capacities_arr, 1.0)
            # Resource scarcity per activity
            scarcity = np.sum(demands_arr / safe_cap, axis=1)
        else:
            scarcity = np.zeros(n, dtype=float)
    else:
        scarcity = np.zeros(n, dtype=float)

    # Successor count
    succ_count = np.array([len(s) for s in succs], dtype=float)

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return np.asarray(durations, dtype=float)

    # Normalize helper
    def normalize(x):
        if len(x) == 0:
            return x
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    tf_norm = normalize(-total_float)
    sc_norm = normalize(scarcity)
    succ_norm = normalize(succ_count)
    dur_norm = normalize(durations)

    # --- Simulation-based bottleneck detection ---
    # Run a few serial SGS simulations with different priority rules
    # Track which activities are "critical" (determine makespan or cause resource conflicts)
    
    bottleneck_score = np.zeros(n, dtype=float)
    n_sims = 0
    
    # Define several simple priority rules for simulation
    def rule_min_float():
        return -total_float  # lower float = higher priority
    
    def rule_max_duration():
        return durations
    
    def rule_max_scarcity():
        return scarcity
    
    def rule_max_succ():
        return succ_count
    
    def rule_combined():
        return 0.5 * (-total_float) + 0.3 * scarcity + 0.2 * succ_count
    
    rules = [rule_min_float, rule_max_duration, rule_max_scarcity, rule_max_succ, rule_combined]
    
    # For each rule, simulate serial SGS and identify bottleneck activities
    # A bottleneck is an activity that, when started, causes resource conflicts
    # or is on the path determining the final makespan
    
    for rule_fn in rules:
        if time.time() - start_time > DEADLINE:
            break
        
        # Simulate serial SGS
        scheduled = [False] * n
        start_times = np.full(n, -1.0)
        end_times = np.full(n, -1.0)
        current_time = 0.0
        remaining_in_deg = in_degree.copy()
        
        # Priority scores for this rule
        prio = np.asarray(rule_fn(), dtype=float)
        
        # Resource usage tracking: for each resource, list of (end_time, activity)
        if demands_arr is not None and capacities_arr is not None:
            n_resources = capacities_arr.shape[0] if capacities_arr.ndim > 0 else 1
            resource_usage = [[] for _ in range(n_resources)]
        else:
            resource_usage = None
        
        n_scheduled = 0
        
        while n_scheduled < n:
            # Find eligible activities (all predecessors scheduled)
            eligible = []
            for i in range(n):
                if not scheduled[i] and remaining_in_deg[i] == 0:
                    eligible.append(i)
            
            if not eligible:
                # Shouldn't happen in a valid DAG, but handle gracefully
                break
            
            # Pick highest priority eligible activity
            best = max(eligible, key=lambda i: prio[i])
            
            # Find earliest feasible start time considering resources
            earliest_start = current_time
            
            if resource_usage is not None:
                for r in range(len(resource_usage)):
                    # Check if this activity's demand on resource r fits
                    if demands_arr[best, r] > 0:
                        # Find when enough capacity is available
                        # Sort by end time
                        resource_usage[r].sort(key=lambda x: x[0])
                        # Check current usage
                        current_usage = sum(amt for (end, amt) in resource_usage[r] if end > earliest_start)
                        # Actually, we need to find the earliest time when usage drops below capacity
                        # Simplified: just check if we can start at earliest_start
                        # For simplicity, we'll just set earliest_start to max of all end times
                        # that conflict with our demand
                        for (end, amt) in resource_usage[r]:
                            if end > earliest_start:
                                earliest_start = max(earliest_start, end)
                        # This is a simplification; in reality we'd need to check cumulative usage
            
            start_times[best] = earliest_start
            end_times[best] = earliest_start + durations[best]
            scheduled[best] = True
            n_scheduled += 1
            current_time = earliest_start
            
            # Update resource usage
            if resource_usage is not None:
                for r in range(len(resource_usage)):
                    if demands_arr[best, r] > 0:
                        resource_usage[r].append((end_times[best], demands_arr[best, r]))
            
            # Update in-degree of successors
            for v in succs[best]:
                remaining_in_deg[v] -= 1
        
        # Identify activities on the makespan-determining path
        # An activity is on the critical path if it's the last to finish
        # or if delaying it would delay the project
        if n_scheduled == n:
            makespan = max(end_times)
            # Activities that finish at or near the makespan are critical
            for i in range(n):
                if end_times[i] >= makespan - 1e-6:
                    bottleneck_score[i] += 1.0
                elif end_times[i] >= makespan - durations[i] * 0.5:
                    bottleneck_score[i] += 0.5
            n_sims += 1
    
    # Normalize bottleneck score
    if n_sims > 0:
        bottleneck_score /= n_sims
    bottleneck_norm = normalize(bottleneck_score)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        # Return static heuristic only
        priority_scores = (0.50 * tf_norm + 0.25 * sc_norm +
                           0.15 * succ_norm + 0.10 * dur_norm)
        return priority_scores

    # Final weighted combination
    # Blend static heuristic with simulation-based bottleneck detection
    w_tf = 0.40
    w_sc = 0.20
    w_succ = 0.10
    w_dur = 0.10
    w_bn = 0.20  # weight for simulation-based bottleneck score

    priority_scores = (w_tf * tf_norm + w_sc * sc_norm +
                       w_succ * succ_norm + w_dur * dur_norm + w_bn * bottleneck_norm)
    
    # Apply mild nonlinearity to amplify differences
    if n > 1:
        min_score = float(np.min(priority_scores))
        max_score = float(np.max(priority_scores))
        if max_score - min_score > 1e-12:
            normalized = (priority_scores - min_score) / (max_score - min_score)
            # Slight power transform to emphasize top activities
            priority_scores = normalized ** 0.95 + 0.05 * tf_norm

    return priority_scores
# EVOLVE-BLOCK-END
