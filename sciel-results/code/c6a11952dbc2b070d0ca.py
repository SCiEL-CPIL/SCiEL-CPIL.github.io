import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # 1. Generate all weight-4 vectors
    candidates = []
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)

    # 2. Compute Conflict Matrix
    G = candidates @ candidates.T
    conflict_mat = G > 2
    
    # 3. Fast Greedy with Masking
    # We try multiple random orders
    
    best_count = 0
    best_selected = []
    
    rng = np.random.default_rng(42)
    
    # Number of restarts depends on time
    # Each restart is O(N^2) ~ 28M ops. 
    # 28M simple ops in NumPy might take ~0.1-0.5s?
    # Let's assume 1s per restart. We can do ~100 restarts.
    
    for i in range(200):
        if time.time() > deadline - 10:
            break
            
        # Random order
        order = rng.permutation(n_cand)
        
        blocked = np.zeros(n_cand, dtype=bool)
        selected = []
        
        for v in order:
            if not blocked[v]:
                selected.append(v)
                # Block all neighbors of v
                blocked[conflict_mat[v]] = True
                
        count = len(selected)
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
            
    if not best_selected:
        return np.array([np.ones(D, dtype=np.int64)], dtype=np.int64)
        
    final_vecs = candidates[best_selected]
    
    # Verify
    norms_sq = np.sum(final_vecs * final_vecs, axis=1)
    max_norm_sq = np.max(norms_sq)
    G_final = final_vecs @ final_vecs.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * G_final
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    
    if min_dist_sq < max_norm_sq:
        # Fallback
        w2 = []
        for comb in itertools.combinations(range(D), 2):
             for signs in itertools.product([1, -1], repeat=2):
                 v = np.zeros(D, dtype=np.int64)
                 for idx, s in zip(comb, signs):
                     v[idx] = s
                 w2.append(v)
        return np.array(w2, dtype=np.int64)

    return final_vecs

# EVOLVE-BLOCK-END
