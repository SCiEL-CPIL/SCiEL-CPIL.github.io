"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def priority(instance):
    """
    Computes priority scores for assembly tasks to minimize the number of stations.
    Strategy: Longest Path to End (LPTE) combined with Task Time.
    This prioritizes tasks that are part of the critical path (high downstream work)
    and breaks ties by processing time (LPT).
    """
    start_time = time.time()
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor list
    # predecessors is likely a list of lists, where predecessors[i] is a list of task indices
    # that must be completed before i.
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
    
    # Calculate Longest Path to End (LPTE)
    # lp[i] = times[i] + max(lp[j] for j in successors[i])
    # If no successors, lp[i] = times[i]
    
    lp = [0.0] * n
    visited = [False] * n
    
    # Use iterative DFS or recursion with memoization
    # Given N=30, recursion is safe, but we'll use a simple stack-based approach or just recursion with increased limit.
    sys.setrecursionlimit(10000)
    
    def compute_lp(i):
        if visited[i]:
            return lp[i]
        
        # Mark as visiting to handle cycles? SALB graphs are DAGs, so no cycles.
        max_succ_lp = 0.0
        for j in successors[i]:
            succ_lp = compute_lp(j)
            if succ_lp > max_succ_lp:
                max_succ_lp = succ_lp
        
        lp[i] = times[i] + max_succ_lp
        visited[i] = True
        return lp[i]
    
    for i in range(n):
        compute_lp(i)
        
    # Deadline Guard
    if time.time() - start_time > 110:
        # Fallback to LPT if something went wrong or took too long (unlikely)
        return times
    
    # Combine LPTE and Time
    # Primary: LPTE (Criticality)
    # Secondary: Time (Size)
    # We want higher score = higher priority.
    # To ensure LPTE dominates, we can scale it or just add them.
    # Since LPTE >= Time, adding them reinforces LPTE.
    # score = LPTE + Time
    
    scores = np.array(lp, dtype=float) + times
    
    # Optional: Check deadline again
    if time.time() - start_time > 115:
         return scores
         
    return scores

# EVOLVE-BLOCK-END
