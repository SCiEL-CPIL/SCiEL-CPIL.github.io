"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    We use a vectorized greedy approach on candidate vectors with norm^2 = 2 and 4.
    The key constraint: for vectors u, v with max_norm_sq = 4,
    we need ||u-v||^2 >= 4, i.e., ||u||^2 + ||v||^2 - 2<u,v> >= 4.
    
    For norm^2=2 vectors: 2+2-2<u,v> >= 4 → <u,v> <= 0
    For norm^2=4 vectors: 4+4-2<u,v> >= 4 → <u,v> <= 2
    For mixed (norm^2=2, norm^2=4): 2+4-2<u,v> >= 4 → <u,v> <= 1
    """
    # Generate candidate vectors
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
    
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute squared norms
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = 4  # Maximum possible norm^2 in our set
    
    # Compute pairwise dot products (vectorized)
    # This is an n x n matrix, which for ~5500 vectors is manageable
    dots = candidates @ candidates.T
    
    # Compute squared distances
    # dist_sq[i,j] = norms_sq[i] + norms_sq[j] - 2*dots[i,j]
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Conflict matrix: conflict[i,j] = True if dist_sq[i,j] < max_norm_sq
    # (and i != j)
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)
    
    # Greedy selection: repeatedly pick a vector with fewest conflicts among remaining
    np.random.seed(42)
    remaining = set(range(n))
    selected = []
    
    while remaining:
        # For each remaining vector, count conflicts with other remaining vectors
        rem_list = list(remaining)
        rem_array = np.array(rem_list)
        
        # Count conflicts for each remaining vector
        conflict_counts = conflict[np.ix_(rem_array, rem_array)].sum(axis=1)
        
        # Pick the one with minimum conflicts (random tie-breaking)
        min_conf = np.min(conflict_counts)
        min_indices = np.where(conflict_counts == min_conf)[0]
        chosen_local = np.random.choice(min_indices)
        chosen_global = rem_list[chosen_local]
        
        selected.append(chosen_global)
        remaining.remove(chosen_global)
        
        # Remove all vectors that conflict with chosen
        conflicts_with_chosen = np.where(conflict[chosen_global])[0]
        to_remove = [c for c in conflicts_with_chosen if c in remaining]
        for c in to_remove:
            remaining.discard(c)
    
    result = candidates[selected]
    return result
# EVOLVE-BLOCK-END
