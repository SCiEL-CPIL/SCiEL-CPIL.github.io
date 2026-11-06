"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    # Build successor list and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        in_degree[i] = len(preds)
        for p in preds:
            successors[p].append(i)
    
    # Topological sort (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_degree_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree_copy[v] -= 1
            if in_degree_copy[v] == 0:
                queue.append(v)
    
    # Compute Rank: longest path from task to end (including own time)
    # Process in reverse topological order
    rank = np.zeros(n)
    for u in reversed(topo_order):
        if successors[u]:
            max_succ_rank = max(rank[v] for v in successors[u])
            rank[u] = times[u] + max_succ_rank
        else:
            rank[u] = times[u]
            
    # Compute Weighted Successor Score: sum of ranks of direct successors
    # This captures the "total criticality" of downstream tasks
    wsc = np.zeros(n)
    for u in reversed(topo_order):
        if successors[u]:
            wsc[u] = sum(rank[v] for v in successors[u])
            
    # Compute Positional Rank: sum of ranks of direct predecessors
    # This captures how much critical work precedes the task
    pr = np.zeros(n)
    for u in topo_order:
        preds = predecessors[u] if u < len(predecessors) else []
        if preds:
            pr[u] = sum(rank[p] for p in preds)
            
    # Normalize components
    def normalize(x):
        if x.max() > x.min():
            return (x - x.min()) / (x.max() - x.min())
        else:
            return np.zeros_like(x)
            
    rank_norm = normalize(rank)
    wsc_norm = normalize(wsc)
    pr_norm = normalize(pr)
    
    # Combine scores
    # Primary: Rank (critical path length)
    # Secondary: WSC (total downstream criticality) - better bottleneck detector than count
    # Tertiary: PR (upstream criticality) - helps break ties for convergence tasks
    # Weights are tuned to favor Rank, then WSC.
    priority_scores = rank_norm + 0.2 * wsc_norm + 0.1 * pr_norm
    
    return priority_scores
# EVOLVE-BLOCK-END
