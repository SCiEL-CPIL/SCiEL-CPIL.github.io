"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy: Generate all integer vectors with squared norm 2 and 4, then use
    a vectorized greedy algorithm to select a maximum-size subset satisfying
    the distance constraint. We use multiple random restarts to find a good packing.
    """
    # Generate candidate vectors: all integer vectors with squared norm 2 or 4
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s * 2
            candidates.append(v)
    
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute squared norms
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Precompute the full distance matrix (squared distances)
    # dist_sq[i,j] = ||c_i - c_j||^2 = norms_sq[i] + norms_sq[j] - 2 * dot(c_i, c_j)
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Conflict matrix: True if dist_sq[i,j] < max_norm_sq (i.e., they conflict)
    conflict = dist_sq < max_norm_sq
    np.fill_diagonal(conflict, False)
    
    # Vectorized greedy with random restarts
    best_selected = []
    np.random.seed(42)
    
    for restart in range(20):
        # Random order of candidates
        order = np.random.permutation(n)
        
        selected = []
        selected_mask = np.zeros(n, dtype=bool)
        
        for idx in order:
            # Check if idx conflicts with any already selected
            if not np.any(selected_mask & conflict[idx]):
                selected.append(idx)
                selected_mask[idx] = True
        
        if len(selected) > len(best_selected):
            best_selected = selected
    
    result = candidates[best_selected]
    return result
# EVOLVE-BLOCK-END
