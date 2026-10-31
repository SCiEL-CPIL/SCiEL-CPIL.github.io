"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    We use a greedy approach on candidate vectors with norm^2 = 8.
    The constraint for norm^2=8 vectors is ||u-v||^2 >= 8, which implies
    <u,v> <= 4.
    
    Candidates are all integer vectors with norm^2 = 8.
    Types:
    1. Two entries are +/-2, rest 0.
    2. Eight entries are +/-1, rest 0.
    """
    candidates = []
    
    # Type 1: Two +/-2's
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si * 2
                v[j] = sj * 2
                candidates.append(v)
                
    # Type 2: Eight +/-1's
    # To avoid generating 42k vectors one by one with Python loops, 
    # we can use itertools but it might be slow. 
    # However, 42k is small enough.
    for positions in itertools.combinations(range(D), 8):
        for signs in itertools.product([1, -1], repeat=8):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Greedy selection
    # We want to select a subset S such that for all u, v in S, <u,v> <= 4.
    # We use a simple greedy with random restarts.
    
    np.random.seed(42)
    best_selected = []
    best_size = 0
    
    # To speed up, we can pre-compute dot products in batches or use a heuristic.
    # Given N~42k, a full N x N matrix is too big for int64 (1.8 GB).
    # But we can do a greedy pass.
    
    # Heuristic: Start with Type 1 vectors (220 of them) as they are more "sparse" 
    # and likely have lower mutual dot products on average?
    # Actually, let's just do a few random shuffles and greedy picks.
    
    for attempt in range(10):
        indices = np.random.permutation(n)
        selected = []
        selected_vectors = []
        
        for idx in indices:
            v = candidates[idx]
            
            # Check against selected vectors
            # If selected is small, this is fast.
            # If selected is large, we can use vectorized dot product.
            if len(selected_vectors) == 0:
                selected.append(idx)
                selected_vectors.append(v)
            else:
                sel_arr = np.array(selected_vectors)
                dots = sel_arr @ v
                if np.max(dots) <= 4:
                    selected.append(idx)
                    selected_vectors.append(v)
                    
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
            
    result = candidates[best_selected]
    return result
# EVOLVE-BLOCK-END
