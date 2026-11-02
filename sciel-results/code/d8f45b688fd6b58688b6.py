"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit
    
    # Generate all weight-4 vectors with ±1 entries
    # C(11,4) = 330 positions, 2^4 = 16 sign combos -> 5280 vectors
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Precompute conflict matrix
    # Conflict if u·v > 2
    G = np.dot(candidates, candidates.T)
    conflict = G > 2
    
    # Self-conflict is not an issue for independent set logic if we handle removal correctly,
    # but strictly, a node doesn't conflict with itself in MIS definition usually,
    # but we remove it anyway.
    np.fill_diagonal(conflict, False)
    
    def min_degree_greedy(seed):
        """
        Min-degree greedy for Maximum Independent Set.
        Randomly breaks ties to allow different runs.
        """
        np.random.seed(seed)
        active = np.ones(n_candidates, dtype=bool)
        # Initial degrees
        degrees = np.sum(conflict, axis=1).astype(np.int32)
        selected_indices = []
        
        indices = np.arange(n_candidates)
        
        while np.any(active):
            # Find active node with minimum degree
            # Set inactive degrees to inf
            degs = np.where(active, degrees, np.inf)
            
            # Find all min values
            min_val = np.min(degs)
            if np.isinf(min_val):
                break
                
            # Get indices with min degree
            candidates_idx = indices[degs == min_val]
            
            # Randomly pick one
            min_idx = candidates_idx[np.random.randint(len(candidates_idx))]
            
            selected_indices.append(min_idx)
            
            # Update degrees for neighbors
            # Neighbors of min_idx that are active
            neighbors = conflict[min_idx, :]
            active_neighbors = neighbors & active
            degrees[active_neighbors] -= 1
            
            # Deactivate min_idx and its neighbors
            to_deactivate = neighbors | (indices == min_idx)
            active[to_deactivate] = False
            
        return selected_indices
    
    best_selected = []
    
    # Run multiple trials of Min-Degree Greedy
    num_trials = 1000 # Can run many because it's fast
    
    for i in range(num_trials):
        if time.time() > deadline:
            break
        
        sel_idx = min_degree_greedy(i)
        if len(sel_idx) > len(best_selected):
            best_selected = sel_idx
            # Optional: break if we reach known optimum 56?
            # But we don't know for sure if 56 is max for this subset.
            # Kissing number in R11 is 56.
            if len(best_selected) >= 56:
                break
    
    # Fallback if no selection found (should not happen)
    if not best_selected:
        best_selected = [0]
        
    return candidates[best_selected]
# EVOLVE-BLOCK-END
