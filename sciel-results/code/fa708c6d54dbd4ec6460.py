import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit
    
    # 1. Generate a large, diverse pool of candidate integer vectors.
    # We want vectors with small squared norms to allow packing more points.
    # Norms considered: 1, 2, 3.
    candidates = []
    
    # Norm 1: ±e_i (11 * 2 = 22 vectors)
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 1
        candidates.append(v)
        candidates.append(-v)
        
    # Norm 2: ±e_i ± e_j (11*10/2 * 4 = 220 vectors)
    for i in range(d):
        for j in range(i + 1, d):
            # +1, +1
            v1 = np.zeros(d, dtype=np.int64)
            v1[i] = 1; v1[j] = 1
            candidates.append(v1)
            candidates.append(-v1)
            
            # +1, -1
            v2 = np.zeros(d, dtype=np.int64)
            v2[i] = 1; v2[j] = -1
            candidates.append(v2)
            candidates.append(-v2)
            
            # -1, +1
            v3 = np.zeros(d, dtype=np.int64)
            v3[i] = -1; v3[j] = 1
            candidates.append(v3)
            candidates.append(-v3)
            
            # -1, -1
            v4 = np.zeros(d, dtype=np.int64)
            v4[i] = -1; v4[j] = -1
            candidates.append(v4)
            candidates.append(-v4)
            
    # Norm 3: ±e_i ± e_j ± e_k (11*10*9/6 * 8 = 1320 vectors)
    # This adds significant diversity.
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            candidates.append(v)
                            candidates.append(-v)
                            
    # We can also add some random vectors with small entries to break symmetries
    # but the structured ones are likely sufficient and faster to generate.
    # Let's add a few random ones just in case.
    for _ in range(1000):
        v = np.zeros(d, dtype=np.int64)
        # Pick 1 to 3 non-zero entries
        num_nonzero = random.randint(1, 3)
        indices = random.sample(range(d), num_nonzero)
        for idx in indices:
            v[idx] = random.choice([-1, 1])
        # Ensure it's not already there (unlikely but good practice)
        # And ensure norm is small (it is, max 3)
        candidates.append(v)
        candidates.append(-v)

    # Remove duplicates and zero vector
    unique_candidates = []
    seen = set()
    for v in candidates:
        if np.all(v == 0):
            continue
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(v)
            
    candidates = np.array(unique_candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Best solution found so far
    best_set = []
    best_score = 0
    
    # Greedy search with multiple restarts and randomization
    # The condition is: min_dist_sq >= max_norm_sq
    # dist_sq(u, v) = ||u||^2 + ||v||^2 - 2 u.v
    # We need ||u||^2 + ||v||^2 - 2 u.v >= max_norm_in_set
    
    max_iter = 50  # Number of restarts
    current_iter = 0
    
    while current_iter < max_iter:
        if time.time() > deadline:
            break
            
        # Shuffle candidates to get different greedy outcomes
        indices = np.arange(n_cand)
        np.random.shuffle(indices)
        shuffled_cands = candidates[indices]
        shuffled_norms = norms_sq[indices]
        
        current_set = []
        current_max_norm = 0
        
        for idx in range(n_cand):
            if time.time() > deadline:
                break
                
            v = shuffled_cands[idx]
            v_norm_sq = shuffled_norms[idx]
            
            # If we add v, the new max norm will be max(current_max_norm, v_norm_sq)
            new_max_norm = max(current_max_norm, v_norm_sq)
            
            # Check compatibility with all points in current_set
            compatible = True
            for u in current_set:
                # dist_sq = ||u||^2 + ||v||^2 - 2 u.v
                # We need dist_sq >= new_max_norm
                # ||u||^2 is stored? We can recompute or store. 
                # For speed, let's store (vector, norm_sq) in current_set
                pass # We need to restructure current_set to hold norms
                
        # Restructure: current_set holds (vector, norm_sq)
        current_set = []
        current_max_norm = 0
        
        for idx in range(n_cand):
            if time.time() > deadline:
                break
                
            v = shuffled_cands[idx]
            v_norm_sq = shuffled_norms[idx]
            
            new_max_norm = max(current_max_norm, v_norm_sq)
            
            compatible = True
            for u, u_norm_sq in current_set:
                dist_sq = u_norm_sq + v_norm_sq - 2 * np.dot(u, v)
                if dist_sq < new_max_norm:
                    compatible = False
                    break
                    
            if compatible:
                current_set.append((v, v_norm_sq))
                current_max_norm = new_max_norm
                
        # Update best
        if len(current_set) > best_score:
            best_score = len(current_set)
            best_set = [v for v, _ in current_set]
            
        current_iter += 1
        
    # If we found a valid set with >= 2 points, return it
    if len(best_set) >= 2:
        result = np.array(best_set, dtype=np.int64)
        # Final verification
        max_norm = np.max(np.sum(result**2, axis=1))
        valid = True
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                d2 = np.dot(result[i] - result[j], result[i] - result[j])
                if d2 < max_norm:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return result
        else:
            # Fallback to a known valid small set if verification fails (should not happen)
            return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    else:
        # Absolute fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
