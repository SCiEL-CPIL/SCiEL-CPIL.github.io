import numpy as np
import itertools
import time

D = 11

def generate_candidates(K):
    candidates = []
    if K == 0:
        return np.array([[]], dtype=np.int64) # Should not happen
    
    # Iterate over possible number of non-zero elements m
    # Max m is K (all 1s)
    for m in range(1, min(K, D) + 1):
        for indices in itertools.combinations(range(D), m):
            # Generate all integer vectors of length m with sum of squares K
            # and no zeros
            def gen_vals(idx, rem_K, curr):
                if len(idx) == 0:
                    if rem_K == 0:
                        yield curr
                    return
                i = idx[0]
                rest = idx[1:]
                max_v = int(np.sqrt(rem_K))
                for v in range(-max_v, max_v + 1):
                    if v == 0:
                        continue
                    sq = v * v
                    if sq > rem_K:
                        continue
                    for rest_vals in gen_vals(rest, rem_K - sq, curr + [v]):
                        yield rest_vals

            for vals in gen_vals(indices, K, []):
                v = np.zeros(D, dtype=np.int64)
                for idx, val in zip(indices, vals):
                    v[idx] = val
                candidates.append(v)
                
    if not candidates:
        return np.empty((0, D), dtype=np.int64)
    return np.array(candidates, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_set = []
    best_score = 0
    
    # Try K from 1 to 5
    for K in [1, 2, 3, 4, 5]:
        if time.time() > deadline:
            break
            
        candidates = generate_candidates(K)
        n_cands = len(candidates)
        if n_cands == 0:
            continue
            
        T = K // 2
        best_for_K = []
        
        # Determine number of passes based on time and candidate count
        # Smaller K -> more passes
        # Larger K -> fewer passes
        # Base passes: 200
        # Adjust: if n_cands > 5000, reduce passes to 50
        # if n_cands > 10000, reduce passes to 20
        
        if n_cands < 500:
            num_passes = 200
        elif n_cands < 5000:
            num_passes = 100
        else:
            num_passes = 50
            
        for p in range(num_passes):
            if time.time() > deadline:
                break
                
            # Shuffle candidates
            # Using np.random.permutation is faster than shuffle on large arrays?
            # Or just shuffle indices
            order = np.random.permutation(n_cands)
            
            current_set = []
            current_arr = np.empty((0, D), dtype=np.int64)
            
            for i in order:
                v = candidates[i]
                
                if len(current_set) == 0:
                    current_set.append(v)
                    current_arr = v[None, :]
                    continue
                    
                # Check compatibility
                dots = current_arr @ v
                if np.any(dots > T):
                    continue
                
                # Compatible
                current_set.append(v)
                # Update current_arr
                # vstack is a bit slow, but acceptable for D=11
                current_arr = np.vstack([current_arr, v[None, :]])
                
            if len(current_set) > len(best_for_K):
                best_for_K = current_set
                
        if len(best_for_K) > best_score:
            best_score = len(best_for_K)
            best_set = best_for_K
            
    if not best_set:
        # Fallback to a single vector
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        return v[None, :]
        
    return np.array(best_set, dtype=np.int64)
