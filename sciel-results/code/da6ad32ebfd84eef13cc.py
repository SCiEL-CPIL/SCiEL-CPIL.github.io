import numpy as np
import itertools
import time

D = 11

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Fallback: The known good 220-vector solution (Norm squared 2)
    def get_seed_solution():
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    seed_sol = get_seed_solution()
    best_k = len(seed_sol)
    best_vecs = seed_sol
    
    # Try to find a better solution using Norm squared 3 vectors
    # Total candidates: C(11,3) * 2^3 = 165 * 8 = 1320
    candidates = []
    for comb in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    
    # Greedy Independent Set
    # We want max set such that min_dist_sq >= max_norm_sq
    # For N=3, max_norm_sq = 3.
    # Condition: ||u-v||^2 >= 3
    # ||u||^2 + ||v||^2 - 2u.v >= 3
    # 3 + 3 - 2u.v >= 3
    # 3 >= 2u.v
    # u.v <= 1.5 -> u.v <= 1
    
    # We run a few random shuffles to find a good independent set
    rng = np.random.default_rng(42)
    
    max_iters = 10
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        perm = rng.permutation(len(candidates))
        shuffled = candidates[perm]
        
        selected = []
        selected_arr = []
        
        for v in shuffled:
            # Check if v conflicts with any in selected
            # Conflict if u.v > 1
            if selected_arr:
                # Compute dot products
                dots = np.dot(selected_arr, v)
                if np.any(dots > 1):
                    continue
            
            selected.append(v)
            selected_arr.append(v)
            
            # Early exit if we already beat the seed
            if len(selected) > best_k:
                # We can break or continue to maximize?
                # Continuing might find even more, but takes time.
                # Let's continue for a bit, but keep the current best.
                pass
        
        if len(selected) > best_k:
            best_k = len(selected)
            best_vecs = np.array(selected, dtype=np.int64)
            
    return best_vecs
