"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor graph
    successors = defaultdict(list)
    
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            
    # Calculate Longest Path to Sink (L_i)
    # We use DFS with memoization as it's efficient for n=30
    memo = {}
    
    def get_longest_path(i):
        if i in memo:
            return memo[i]
        
        max_path = 0.0
        for s in successors[i]:
            p = get_longest_path(s)
            if p > max_path:
                max_path = p
                
        res = times[i] + max_path
        memo[i] = res
        return res

    # Calculate Number of Immediate Successors (NS_i)
    # This is simply the length of the successors list
    
    priorities = np.zeros(n, dtype=float)
    for i in range(n):
        l_i = get_longest_path(i)
        ns_i = len(successors[i])
        # Combination Heuristic: Longest Path + Successor Count
        # We add successor count to encourage tasks that unlock more downstream tasks
        priorities[i] = l_i + ns_i
        
    return priorities
# EVOLVE-BLOCK-END
