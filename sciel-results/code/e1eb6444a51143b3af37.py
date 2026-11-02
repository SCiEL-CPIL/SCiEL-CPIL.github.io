import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # 1. Generate Candidates
    candidates = []
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int8)
    N = len(candidates)
    
    # 2. Conflict Matrix
    M = candidates @ candidates.T
    conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Adjacency list for faster neighbor lookup?
    # Or just use the matrix.
    
    best_size = 0
    best_solution = np.zeros(N, dtype=bool)
    rng = np.random.default_rng(42)
    
    n_restarts = 100
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
            
        selected = np.zeros(N, dtype=bool)
        available = list(range(N))
        rng.shuffle(available)
        
        count = 0
        
        # Greedy with list maintenance
        while available and time.time() < deadline - 5:
            # Pick first available
            idx = available.pop(0)
            
            # Check if it conflicts with any selected
            # Since we remove neighbors from available when we select, 
            # any remaining available node does NOT conflict with selected nodes?
            # NO. We remove neighbors of the SELECTED node from available.
            # So if we pick idx from available, it is not a neighbor of any previously selected node.
            # Thus, it is safe to select it.
            
            selected[idx] = True
            count += 1
            
            # Remove idx and all its neighbors from available
            # Neighbors are those j where conflict[idx, j] == 1
            neighbors = np.where(conflict[idx] > 0)[0]
            # Also remove idx itself (already popped)
            
            # Efficient removal:
            # Create a set of neighbors to remove
            to_remove = set(neighbors)
            
            # Filter available
            # This is O(N) per selection. Total O(N^2).
            # N=5500, N^2=30M. 100 restarts = 3B. Too slow.
            
            # Better: Use a boolean array `in_available`
            pass
            
        # Let's switch to the boolean array approach for speed
        if time.time() > deadline - 5:
            break
            
        selected = np.zeros(N, dtype=bool)
        in_avail = np.ones(N, dtype=bool)
        
        # Random order
        order = rng.permutation(N)
        
        count = 0
        for idx in order:
            if not in_avail[idx]:
                continue
            # Since we remove neighbors when we select, if idx is still available, 
            # it doesn't conflict with any selected node.
            selected[idx] = True
            in_avail[idx] = False
            count += 1
            
            # Remove neighbors
            neighbors = np.where(conflict[idx] > 0)[0]
            in_avail[neighbors] = False
            
        if count > best_size:
            best_size = count
            best_solution = selected.copy()
            
        # Local Search: Try to add more
        # This is tricky with the "in_avail" state because it's a maximal set.
        # We can try to "shake" by removing a few selected nodes and re-running greedy on the freed nodes.
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            selected_indices = np.where(selected)[0]
            if len(selected_indices) == 0:
                break
                
            # Pick a random selected node to remove
            u = selected_indices[rng.integers(len(selected_indices))]
            selected[u] = False
            in_avail[u] = True
            count -= 1
            
            # Also, nodes that were removed ONLY because of u might now be available?
            # No, we don't track "why" they were removed.
            # We just try to add any node that is compatible with current selected set.
            
            # Check all nodes not selected
            unselected = np.where(~selected)[0]
            # This is O(N).
            # For each, check if compatible.
            # Compatibility: not conflict with any selected.
            
            # Vectorized check is hard for "any conflict with selected".
            # We can compute: for each unselected node v, is conflict[v] & selected all zeros?
            # This is (conflict[unselected] @ selected) == 0?
            # conflict is N x N. selected is N.
            # conflict[unselected] is K x N.
            # Product is K x 1.
            
            if len(unselected) > 0:
                # Efficient check
                # dots = conflict[unselected] @ selected
                # But conflict is uint8.
                # We want to know if ANY is 1.
                # sum = np.sum(conflict[unselected] * selected[:, None], axis=1)
                # This is expensive.
                
                # Alternative: Just iterate
                # If N is 5500, and we do this in a loop, it might be slow.
                # But we only do it a few times.
                
                added = 0
                for v in unselected:
                    if not np.any(conflict[v] & selected):
                        selected[v] = True
                        in_avail[v] = False # Keep consistent
                        count += 1
                        added += 1
                        
                if count > best_size:
                    best_size = count
                    best_solution = selected.copy()
                if added > 0:
                    improved = True

    if best_size == 0:
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return candidates[best_solution].astype(np.int64)
