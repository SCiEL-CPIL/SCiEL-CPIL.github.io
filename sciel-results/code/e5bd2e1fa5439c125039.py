"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # D11 lattice: all nonzero vectors with entries in {-1, 0, 1}
    # such that the vector has even weight and even number of -1's
    # This gives the D11 root lattice points
    # All have norm_sq = weight, and minimum distance between distinct points is 2
    
    # Actually, let's use a simpler approach:
    # Use all vectors of type (±1, ±1, 0, ..., 0) - weight 2 vectors
    # These have norm 2, and any two distinct such vectors have distance >= 2
    # Number: C(11,2) * 4 = 55 * 4 = 220
    
    # But we can do better by including weight 4 vectors from D11 lattice
    # D11 lattice: x_i ∈ Z, sum x_i even
    # We want vectors with small norm. Let's use all vectors with entries in {-1,0,1}
    # and even weight (number of nonzero entries is even)
    
    # For weight 2: C(11,2) * 4 = 220 vectors
    # For weight 4: C(11,4) * 16 = 5280 vectors, but not all are in D11
    # In D11, we need sum of coordinates to be even
    # For weight 4 with ±1 entries, sum is even iff number of +1's is even
    # So 8 sign combinations per position set (4 choose 0,2,4 = 1+6+1=8)
    # That gives C(11,4) * 8 = 330 * 8 = 2640 weight-4 vectors
    
    # Let's generate all D11 lattice points with norm_sq <= 4
    candidates = []
    
    # Weight 2 vectors
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                candidates.append(v)
    
    # Weight 4 vectors in D11 (even number of -1's)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            if sum(1 for s in signs if s == -1) % 2 == 0:
                v = np.zeros(D, dtype=np.int64)
                for pos, s in zip(positions, signs):
                    v[pos] = s
                candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Constraint: for vectors with norm_sq = w, we need |u-v|^2 >= max_norm_sq
    # But different vectors may have different norms.
    # To simplify, let's normalize: we want min_dist >= max_norm
    # 
    # For weight-2 vectors: norm = 2
    # For weight-4 vectors: norm = 2
    # All have norm 2! So we need |u-v|^2 >= 4
    # |u-v|^2 = |u|^2 + |v|^2 - 2u·v = 4 + 4 - 2u·v = 8 - 2u·v
    # Need 8 - 2u·v >= 4 => u·v <= 2
    
    # Precompute dot product matrix
    dots = candidates @ candidates.T
    
    # Greedy selection
    rng = np.random.default_rng(42)
    best_set = []
    
    for restart in range(500):
        if time.time() > deadline - 10.0:
            break
        
        order = rng.permutation(n_candidates)
        selected = []
        selected_set = set()
        
        for i in order:
            if time.time() > deadline - 5.0:
                break
            
            if i in selected_set:
                continue
            
            if not selected:
                selected.append(i)
                selected_set.add(i)
                continue
            
            sel_arr = np.array(selected)
            if np.all(dots[i, sel_arr] <= 2):
                selected.append(i)
                selected_set.add(i)
        
        if len(selected) > len(best_set):
            best_set = selected
            if len(best_set) >= 2000:
                break
    
    if not best_set:
        # Fallback: return weight-2 vectors only
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result = candidates[best_set]
    
    # Final verification
    if len(result) > 1:
        from scipy.spatial.distance import cdist
        dists = cdist(result, result)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        max_norm = np.max(np.linalg.norm(result, axis=1))
        
        if min_dist < max_norm - 1e-9:
            # Fallback to weight-2 vectors
            vectors = []
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        vectors.append(v)
            return np.array(vectors, dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END
