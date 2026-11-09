"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = start_time + 10.0  # 10s safety margin within 120s limit
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor list
    # predecessors is typically a list of lists, where predecessors[i] is a list of indices
    # We need to invert this to get successors[i]
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
            
    # Calculate Ranked Positional Weights (RPW)
    # W[i] = times[i] + sum(W[j] for j in successors[i])
    # We need to compute this in reverse topological order.
    # For small N=30, we can just use a simple iterative approach or DFS.
    # Since it's a DAG, we can use a topological sort or simply iterate until convergence 
    # or use a recursive memoized function.
    
    # Let's use a simple topological sort based approach for correctness and speed.
    # Or simpler: since N is small, we can just compute the longest path to the end 
    # using DFS with memoization.
    
    weights = [0.0] * n
    visited = [False] * n
    
    def calc_weight(i):
        if visited[i]:
            return weights[i]
        visited[i] = True
        w = times[i]
        for s in successors[i]:
            w += calc_weight(s)
        weights[i] = w
        return w
    
    for i in range(n):
        calc_weight(i)
        
    # Check deadline
    if time.time() > deadline:
        # Fallback to simple LTF if something went wrong or took too long (unlikely)
        return times
        
    return np.array(weights, dtype=float)
# EVOLVE-BLOCK-END
