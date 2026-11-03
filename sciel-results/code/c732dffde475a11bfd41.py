import numpy as np
from itertools import combinations
import random

def construct():
    D = 11
    all_subsets = list(combinations(range(D), 4))
    n_subsets = len(all_subsets)
    
    # Precompute conflicts
    # conflict[i][j] = True if subset i and j share 3 elements
    # Or just store list of conflicts for each subset
    conflicts = [[] for _ in range(n_subsets)]
    for i in range(n_subsets):
        for j in range(i + 1, n_subsets):
            # Check intersection
            # Since size is 4, intersection 3 means symmetric difference is 1
            # Or just count common elements
            if len(set(all_subsets[i]) & set(all_subsets[j])) >= 3:
                conflicts[i].append(j)
                conflicts[j].append(i)
                
    best_count = 0
    best_selected = []
    
    # Try multiple random orderings or strategies
    for _ in range(50):
        # Strategy: Greedy with "Least Future Conflicts"
        # At each step, pick the available subset that conflicts with the fewest other available subsets
        available = set(range(n_subsets))
        selected = []
        
        while available:
            # Find subset in available with min conflicts within available
            min_conf = n_subsets + 1
            best_s = -1
            # Optimization: This is O(N^2) per step, N=330. 330^2 * 41 steps ~ 4.5M ops. Fast.
            for s in available:
                conf_count = 0
                for c in conflicts[s]:
                    if c in available:
                        conf_count += 1
                if conf_count < min_conf:
                    min_conf = conf_count
                    best_s = s
            
            if best_s == -1:
                break
                
            selected.append(best_s)
            available.remove(best_s)
            # Remove all conflicts of best_s from available
            for c in conflicts[best_s]:
                if c in available:
                    available.remove(c)
                    
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected
            
    # Generate vectors
    vectors = []
    for idx in best_selected:
        subset = all_subsets[idx]
        i, j, k, l = subset
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        vectors.append(v)
                        
    return np.array(vectors, dtype=np.int64)
