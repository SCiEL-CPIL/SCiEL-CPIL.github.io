import numpy as np
import time
from collections import deque


def priority(instance):
    t0 = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Build predecessor lists
    predecessors = instance.get("predecessors", [])
    # predecessors[i] is a list of activity indices that must complete before i starts
    
    # Build successors and in-degree
    in_degree = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # Forward pass: earliest start times (ES)
    es = np.zeros(n, dtype=float)
    for node in topo_order:
        max_pred = 0.0
        preds = predecessors[node] if node < len(predecessors) else []
        for p in preds:
            if es[p] + durations[p] > max_pred:
                max_pred = es[p] + durations[p]
        es[node] = max_pred
    
    # Backward pass: latest start times (LS)
    # First compute project makespan (earliest finish of all activities)
    project_end = max(es[i] + durations[i] for i in range(n)) if n > 0 else 0.0
    
    ls = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        # Latest finish = min of (LS of successors)
        min_succ_ls = project_end
        for s in successors[node]:
            if ls[s] < min_succ_ls:
                min_succ_ls = ls[s]
        ls[node] = min_succ_ls - durations[node]
    
    # Total float (slack) = LS - ES
    slack = ls - es
    
    # Resource demand intensity: sum of (demand_j / capacity_j) for each activity
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    resource_score = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        if demands_arr.ndim == 2:
            safe_cap = np.where(capacities_arr > 0, capacities_arr, 1.0)
            resource_score = np.sum(demands_arr / safe_cap, axis=1)
        else:
            safe_cap = np.where(capacities_arr > 0, capacities_arr, 1.0)
            resource_score = demands_arr / safe_cap
    
    # Deadline check
    if time.time() - t0 > DEADLINE:
        # Return simple critical path priority as fallback
        return -slack.copy()
    
    # Compute backward critical path length (longest path from node to end)
    cp = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ = 0.0
        for s in successors[node]:
            if cp[s] > max_succ:
                max_succ = cp[s]
        cp[node] = durations[node] + max_succ
    
    # Priority: lower slack = higher priority (more urgent)
    # Higher resource demand = higher priority (more constraining)
    # Higher critical path length = higher priority (on critical path)
    
    # Normalize each component to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.zeros_like(arr)
    
    # Invert slack: activities with LESS slack should have HIGHER priority
    slack_inv = 1.0 - normalize(slack)
    
    cp_norm = normalize(cp)
    rs_norm = normalize(resource_score)
    
    # Also consider duration: longer activities on critical path are more important
    dur_norm = normalize(durations)
    
    # Compute a "bottleneck" score: how much this activity's resource demand
    # constrains the project relative to its duration (resource pressure per time unit)
    # Activities that use lots of resources for a short time are more disruptive
    # to the schedule than those that use resources gradually
    if demands is not None and capacities is not None:
        safe_dur = np.where(durations > 0, durations, 1.0)
        resource_pressure = resource_score / safe_dur
        rp_norm = normalize(resource_pressure)
    else:
        rp_norm = np.zeros(n, dtype=float)
    
    # Weighted combination with refined weights:
    # - Slack (total float) is the most important indicator of urgency
    # - Critical path length indicates importance  
    # - Resource pressure (demand/duration) captures scheduling disruption
    # - Resource demand as secondary resource constraint
    # - Duration as minor tiebreaker
    priority_scores = (0.45 * slack_inv + 
                       0.25 * cp_norm + 
                       0.15 * rp_norm + 
                       0.10 * rs_norm + 
                       0.05 * dur_norm)
    
    return priority_scores
