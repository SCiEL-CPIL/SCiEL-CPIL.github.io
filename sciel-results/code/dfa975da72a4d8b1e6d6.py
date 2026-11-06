import numpy as np
from collections import deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    # Build successor lists
    succs = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
            in_degree[i] += 1
            
    # Topological sort to get order
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
            
    topo_order = []
    in_degree_copy = in_degree.copy()
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree_copy[v] -= 1
            if in_degree_copy[v] == 0:
                q.append(v)
                
    # If graph is not a DAG or has issues, fall back to LTF
    if len(topo_order) != n:
        return times

    # Calculate Rank (Longest path to sink)
    # Process in reverse topological order
    rank = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        max_succ_rank = 0.0
        for v in succs[u]:
            if rank[v] > max_succ_rank:
                max_succ_rank = rank[v]
        rank[u] = times[u] + max_succ_rank
        
    # Tie-breaking: 
    # 1. Rank is primary.
    # 2. If ranks are equal, prefer tasks with more successors (unlocks more).
    # 3. If still equal, prefer shorter times (packs better)? Or longer?
    # Usually, for same rank, shorter time is better for fitting in current station.
    # But the greedy algorithm picks highest priority first.
    # Let's add a small epsilon for successors count to break ties.
    succ_counts = np.array([len(s) for s in succs], dtype=float)
    
    # Normalize or just combine
    # Rank is in units of time. Succ count is small.
    # We want high Rank to be high priority.
    # We want high Succ Count to be high priority (tie break).
    
    # To ensure Rank dominates, we can scale succ counts.
    # Max rank could be large. Max succ count is <= n.
    # Let's just use rank. It's the most standard.
    # However, a simple rank might tie often.
    # Let's try: rank + 0.001 * succ_count
    
    scores = rank + 0.001 * succ_counts
    
    return scores
