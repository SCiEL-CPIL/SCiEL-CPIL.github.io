import numpy as np
import itertools

def construct():
    D = 11
    candidates = []
    
    # Type 1: two ±1's
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Type 2: one ±2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    # Type 3: four ±1's
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    dots = candidates @ candidates.T
    conflict = dots > 2
    np.fill_diagonal(conflict, False)
    
    # Adjacency list for faster updates
    # neighbors[i] = list of j such that conflict[i, j]
    neighbors = [np.where(conflict[i])[0] for i in range(n)]
    
    best_size = 0
    best_selected = []
    np.random.seed(42)
    
    for attempt in range(500):
        remaining = np.ones(n, dtype=bool)
        degrees = conflict.sum(axis=1).copy()
        selected = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            # Find min degree among remaining
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            # Update degrees and remaining
            # Remove chosen_global
            remaining[chosen_global] = False
            degrees[chosen_global] = 0
            
            # For each neighbor of chosen_global, if remaining, decrement degree and mark for removal?
            # No, in greedy MIS, we remove the node AND its neighbors.
            # So we don't just decrement, we remove them.
            
            neighs = neighbors[chosen_global]
            to_remove = neighs[remaining[neighs]] # Only those still remaining
            
            for node in to_remove:
                remaining[node] = False
                # No need to update degrees of their neighbors because they are removed
                # But we need to update degrees of remaining nodes that are neighbors of 'node'
                # This is getting complex. 
                # Standard greedy:
                # 1. Pick u.
                # 2. Add u to set.
                # 3. Remove u and N(u) from graph.
                # 4. Repeat.
                
            # To update degrees correctly for the next step:
            # The degrees of the remaining nodes should reflect the new graph.
            # Since we removed a set of nodes R (chosen + neighbors), 
            # the degree of any remaining node v is reduced by the number of its neighbors in R.
            
            # This is O(N) per step if we iterate through R.
            # R size is approx degree(u).
            
            # Let's just recompute degrees for remaining nodes if it's not too slow?
            # No, let's stick to the simple recomputation if N is small enough, 
            # or optimize the removal.
            
            # Given the previous analysis, recomputation is O(K^2).
            # With degree update, it is O(K * deg).
            
            # I will use the simple recomputation for correctness and simplicity, 
            # but limit the number of attempts to avoid timeout.
            # If N=5500, and it takes 5s per run, 10 runs is 50s.
            # So I will run 10-20 attempts.
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
            
    result = candidates[best_selected]
    return result
