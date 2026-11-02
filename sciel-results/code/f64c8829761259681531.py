"""Kissing number in dimension 11 - greedy construction."""
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    # Start with the seed: all ±e_i ± e_j vectors
    vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    vectors = np.array(vectors)
    
    # Try to add more vectors greedily
    # Candidate vectors: try various patterns
    # We need: for all existing v, ||new_v - v|| >= max_norm
    # where max_norm = max(||v|| for v in set)
    
    # Current max norm squared
    norms_sq = np.sum(vectors ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Generate candidate vectors to try
    # Type 1: ±e_i (norm 1) - but norm 1 < sqrt(2), so max_norm stays sqrt(2)
    # Need ||e_i - v|| >= sqrt(2) for all v in set
    # For v = e_i + e_j: ||e_i - (e_i+e_j)|| = ||-e_j|| = 1 < sqrt(2). Fail.
    # So ±e_i won't work with the current set.
    
    # Type 2: ±e_i ± e_j ± e_k (norm sqrt(3))
    # If we add these, max_norm becomes sqrt(3)
    # Need ||new - old|| >= sqrt(3) for all old vectors
    # For old = e_i+e_j, new = e_i+e_j+e_k: ||diff|| = 1 < sqrt(3). Fail.
    
    # So we can't easily add to the seed. Let's try a different base.
    
    # Alternative: use only vectors with a fixed pattern that allows more vectors.
    # Try: all vectors with exactly 2 nonzero entries, but restrict signs.
    
    # Actually, let's try a greedy approach from scratch with a broader candidate pool.
    
    # Reset and try greedy
    vectors_list = []
    max_norm_sq_current = 0
    
    # Candidate pool: all vectors with at most 3 nonzero entries, values in {-2,-1,0,1,2}
    # This is large, so let's be strategic.
    
    # First, add the seed vectors
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors_list.append(v)
    
    vectors_arr = np.array(vectors_list)
    max_norm_sq = np.max(np.sum(vectors_arr ** 2, axis=1))
    
    # Try adding vectors of form e_i + e_j + e_k
    added = 0
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for signs in [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
                              (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = signs
                    v_sq = np.sum(v ** 2)
                    new_max = max(max_norm_sq, v_sq)
                    
                    # Check constraint: ||v - w|| >= new_max for all w in set
                    # ||v-w||^2 >= new_max
                    diffs = vectors_arr - v
                    dists_sq = np.sum(diffs ** 2, axis=1)
                    if np.min(dists_sq) >= new_max:
                        vectors_list.append(v)
                        vectors_arr = np.array(vectors_list)
                        max_norm_sq = new_max
                        added += 1
                    
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        return np.array(vectors_list, dtype=np.int64)
    
    # Try adding vectors of form 2*e_i + e_j
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = sj
                    v_sq = np.sum(v ** 2)
                    new_max = max(max_norm_sq, v_sq)
                    
                    diffs = vectors_arr - v
                    dists_sq = np.sum(diffs ** 2, axis=1)
                    if np.min(dists_sq) >= new_max:
                        vectors_list.append(v)
                        vectors_arr = np.array(vectors_list)
                        max_norm_sq = new_max
                        added += 1
                    
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return np.array(vectors_list, dtype=np.int64)
