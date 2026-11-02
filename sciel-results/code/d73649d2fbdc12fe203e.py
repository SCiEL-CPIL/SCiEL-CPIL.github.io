"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def _generate_candidates(weights):
    """Generate all integer vectors with the specified weights."""
    candidates = []
    for w in weights:
        if w > D:
            continue
        # Choose w indices out of D
        for indices in itertools.combinations(range(D), w):
            # Choose signs for these indices
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for idx, sign in zip(indices, signs):
                    v[idx] = sign
                candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Generate candidates with weight 3 and 4
    # Weight 3: N=3. Condition with other weight 3: dot <= 1.
    # Weight 4: N=4. Condition with other weight 4: dot <= 2.
    # Mixed 3,4: Condition is dist^2 >= max(N1, N2).
    
    candidates = _generate_candidates([3, 4])
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    best_vectors = []
    best_count = 0
    
    # We will perform multiple random restarts of the greedy algorithm
    # To speed up, we can vectorize the dot product check against the current selected set.
    
    max_restarts = 1000
    rng = np.random.default_rng(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle candidates
        perm = rng.permutation(n_cand)
        
        selected_indices = []
        selected_vectors = []
        
        for idx in perm:
            v = candidates[idx]
            n_v = norms_sq[idx]
            
            if not selected_vectors:
                selected_indices.append(idx)
                selected_vectors.append(v)
                continue
            
            # Check compatibility with all selected vectors
            # We need ||v - w||^2 >= max(n_v, n_w) for all w in selected
            # ||v - w||^2 = n_v + n_w - 2 v.dot(w)
            # So n_v + n_w - 2 v.dot(w) >= max(n_v, n_w)
            
            # Vectorized check
            # Get selected vectors as array
            sel_arr = np.array(selected_vectors)
            sel_norms = np.array([norms_sq[i] for i in selected_indices])
            
            # Compute dot products
            dots = sel_arr @ v
            
            # Compute distances squared
            dist_sq = n_v + sel_norms - 2 * dots
            
            # Compute max norms
            max_norms = np.maximum(n_v, sel_norms)
            
            # Check if all distances are sufficient
            if np.all(dist_sq >= max_norms):
                selected_indices.append(idx)
                selected_vectors.append(v)
        
        if len(selected_vectors) > best_count:
            best_count = len(selected_vectors)
            best_vectors = selected_vectors.copy()
            
            # Optional: early stop if we exceed a very high threshold? 
            # Kissing number in 11D is likely > 500. 
            # If we get > 800, it's a great score.
            if best_count > 1000:
                break

    if not best_vectors:
        # Fallback to seed if something went wrong or time was too tight
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return np.array(best_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
