import numpy as np
import itertools
import time

def construct():
    D = 11
    # Build the pool of minimal-norm (norm^2 = 4) integer vectors in R^11
    # These are:
    #   Type A: exactly four ±1's, rest 0
    #   Type B: exactly one ±2, rest 0
    #
    # The Leech lattice's minimal vectors in dimension 11 (from the E8 x A2 x A1
    # decomposition) include all such vectors. The kissing number in dim 11 is
    # known to be at least 504 (and conjectured to be 504). We generate all
    # norm-squared-4 integer vectors and find a large independent set in the
    # conflict graph where two vectors conflict if their distance squared < 4
    # (i.e., they are not at distance >= 2 apart, since max norm = 2).
    
    # Generate Type A: four ±1's
    type_a = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            type_a.append(v)
    type_a = np.array(type_a, dtype=np.int64)
    
    # Generate Type B: one ±2
    type_b = []
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            type_b.append(v)
    type_b = np.array(type_b, dtype=np.int64)
    
    candidates = np.vstack([type_a, type_b])
    n = len(candidates)
    
    norms_sq = np.sum(candidates**2, axis=1)
    # All have norm_sq = 4, so norm = 2
    # Condition: min pairwise dist >= max norm = 2, i.e., dist_sq >= 4
    
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    # Use a greedy independent set algorithm with multiple random restarts
    # We use a degree-based heuristic: always pick the vertex with minimum degree
    # among remaining, with random tie-breaking for diversity.
    
    best_selected = []
    best_size = 0
    np.random.seed(12345)
    
    start_time = time.time()
    max_attempts = 500
    
    for attempt in range(max_attempts):
        if time.time() - start_time > 100:
            break
        
        remaining = np.ones(n, dtype=bool)
        # Compute degrees
        degrees = conflict.sum(axis=1).astype(np.int32).copy()
        selected = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            # Remove chosen and its neighbors
            neigh_mask = conflict[chosen_global]
            to_remove = np.zeros(n, dtype=bool)
            to_remove[chosen_global] = True
            to_remove[neigh_mask] = True
            actual_remove = to_remove & remaining
            
            if np.any(actual_remove):
                # Update degrees: for each remaining vertex, subtract number of removed neighbors
                deg_decrease = conflict[actual_remove][:, remaining].sum(axis=0)
                degrees[remaining] -= deg_decrease
            
            remaining[actual_remove] = False
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
            # Early stop if we hit the known kissing number
            if best_size >= 504:
                break
    
    result = candidates[best_selected]
    return result
