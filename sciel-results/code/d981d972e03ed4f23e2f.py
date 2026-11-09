import numpy as np

def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    n_res = len(capacities)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Compute CPM Rank (Longest path from i to end)
    # We need topological order. 
    # Since predecessors are given, we can assume indices might be topologically sorted?
    # Usually in these tasks, activities are 0..n-1 and predecessors[i] contains indices < i.
    # Let's verify or just do a topological sort to be safe.
    
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
            
    # Simple topological sort (Kahn's algorithm)
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    temp_in_degree = in_degree.copy()
    while queue:
        # Pick a node (order doesn't matter for CPM rank calc if we do it backwards, 
        # but we need the order to compute ranks backwards)
        u = queue.pop(0) 
        topo_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
                
    # If graph is not connected or has cycles, handle gracefully. 
    # Assuming valid DAG.
    
    # Calculate CPM Rank (successor depth)
    cpm_rank = np.zeros(n, dtype=float)
    # Iterate in reverse topological order
    for u in reversed(topo_order):
        if not successors[u]:
            cpm_rank[u] = durations[u]
        else:
            max_succ = max(cpm_rank[v] for v in successors[u])
            cpm_rank[u] = durations[u] + max_succ
            
    # Calculate Resource Pressure
    # pressure[i] = max(demands[i, r] / capacities[r])
    # Handle division by zero if capacity is 0 (shouldn't happen)
    safe_caps = np.where(capacities == 0, 1, capacities)
    ratios = demands / safe_caps
    resource_pressure = np.max(ratios, axis=1)
    
    # Calculate Successor Count
    succ_count = np.array([len(s) for s in successors], dtype=float)
    
    # Calculate Predecessor Count
    pred_count = np.array([len(p) for p in predecessors], dtype=float)
    
    # Combine Scores
    # CPM rank is the most important. 
    # We want higher score = higher priority.
    # CPM rank: higher is better (more critical path remaining).
    # Resource pressure: higher is better (more bottleneck).
    # Succ count: higher is better (more branching, more impact).
    # Pred count: lower is better (more flexibility, fewer constraints) -> so we subtract or use negative weight
    
    # Weights:
    # CPM rank is in time units (e.g., 10-100).
    # Resource pressure is ratio (0-1).
    # Succ count is small (0-5).
    
    # Let's weight them to be comparable.
    # Average duration is roughly sum(durations)/n.
    # Let's use a fixed weight that makes resource pressure significant but secondary.
    # If avg duration is 10, a pressure of 1.0 should add maybe 5-10 to the score.
    
    # Heuristic weights
    w_cpm = 1.0
    w_res = 5.0
    w_succ = 0.5
    w_pred = -0.5
    
    scores = w_cpm * cpm_rank + w_res * resource_pressure + w_succ * succ_count + w_pred * pred_count
    
    return scores
