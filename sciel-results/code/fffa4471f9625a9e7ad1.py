"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Calculates priority scores for assembly tasks using a refined heuristic.
    
    The primary signal is the 'Rank' (Longest Path to end), which identifies tasks
    on the critical path. Secondary signals include total downstream work and
    connectivity to break ties and identify bottlenecks that rank alone might miss.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    times = np.array(instance["times"], dtype=float)
    n_tasks = len(times)
    preds = instance["predecessors"]
    
    # Build adjacency list for successors
    successors = [[] for _ in range(n_tasks)]
    in_degree = [0] * n_tasks
    
    for i in range(n_tasks):
        if i < len(preds):
            for p in preds[i]:
                # p is a predecessor of i, so i is a successor of p
                successors[p].append(i)
                in_degree[i] += 1
                
    # Topological Sort (Kahn's Algorithm)
    queue = deque()
    indeg = in_degree[:]
    for i in range(n_tasks):
        if indeg[i] == 0:
            queue.append(i)
            
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
                
    # If graph is cyclic or malformed, topo_order might be incomplete. Assume valid DAG.
    # If not all tasks are in topo_order, we need a fallback or handle it.
    # For now, assume valid DAG as per task description.
    
    # Calculate Rank (Longest Path) and Total Downstream Time in reverse topological order
    rank = np.zeros(n_tasks, dtype=float)
    total_down = np.zeros(n_tasks, dtype=float)
    
    # Iterate backwards through topological order
    for i in reversed(topo_order):
        # For rank: longest path through successors
        max_succ_rank = 0.0
        # For total_down: sum of all downstream work
        sum_succ_down = 0.0
        
        if successors[i]:
            succ_ranks = [rank[j] for j in successors[i]]
            max_succ_rank = max(succ_ranks)
            succ_downs = [total_down[j] for j in successors[i]]
            sum_succ_down = sum(succ_downs)
        
        rank[i] = times[i] + max_succ_rank
        total_down[i] = times[i] + sum_succ_down
        
    # Secondary features
    num_succ = np.array([len(successors[i]) for i in range(n_tasks)], dtype=float)
    
    # Check deadline
    if time.time() > deadline:
        # Return a simple fallback if we ran out of time (unlikely for n=30)
        return times.copy()
    
    # Composite Score:
    # Primary: Rank (critical path) - most important for minimizing stations
    # Secondary: Total downstream time - identifies true bottlenecks
    # Tertiary: Number of successors - breaks ties favoring high-connectivity tasks
    
    # Weights are tuned to emphasize the critical path while using downstream
    # work to distinguish between tasks with similar ranks.
    # Rank and Total Downstream are both in time units.
    # Num Succ is a count.
    
    # Normalization is not strictly necessary if scales are comparable, but
    # we can use weights to balance them.
    # Rank is typically the dominant factor.
    
    scores = 1.0 * rank + 0.15 * total_down + 0.05 * num_succ
    
    return scores

# EVOLVE-BLOCK-END
