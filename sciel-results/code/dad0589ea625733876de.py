import numpy as np
import itertools
import time

D = 11

def generate_candidates():
    candidates = []
    # Weight 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Weight 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                candidates.append(v)
                
    # Weight 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
                    
    # Weight 4
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append(v)
                        
    return np.array(candidates)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Safety margin
    
    candidates = generate_candidates()
    n_candidates = len(candidates)
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    best_vectors = []
    
    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        selected_indices = []
        selected_norms_sq = []
        
        # Shuffle candidates
        indices = np.arange(n_candidates)
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            v = candidates[idx]
            v_norm_sq = norms_sq[idx]
            
            if len(selected_indices) == 0:
                selected_indices.append(idx)
                selected_norms_sq.append(v_norm_sq)
                continue
                
            # Check compatibility
            # selected_vectors is a view? No, let's keep it as list of arrays or matrix.
            # For efficiency, we can maintain a matrix of selected vectors.
            # But appending to a list and converting is O(K).
            # Let's use a list of arrays for simplicity, and vectorize dot product.
            
            # Actually, maintaining a matrix is better for vectorization.
            # But we are in a loop.
            # Let's just do the check.
            
            # If we have many selected, this gets slow.
            # Let's optimize:
            # We can break early if incompatible.
            
            compatible = True
            # Get selected vectors
            # To avoid creating a large array every time, we can just loop?
            # No, vectorized is faster for K > 10.
            
            # Let's keep selected as a list of indices and use numpy operations on the subset.
            # This requires gathering rows.
            # sel_vecs = candidates[selected_indices]
            # This is O(K*D).
            
            # If K is small, it's fine. If K is large (500), 500*11 = 5500 ops.
            # n_candidates is 6800.
            # Total ops ~ 6800 * 500 * 11 ~ 37 million.
            # In Python with Numpy, this should be fast (< 1s per restart).
            
            sel_vecs = candidates[selected_indices]
            sel_norms = norms_sq[selected_indices]
            
            dots = sel_vecs @ v
            dist_sqs = v_norm_sq + sel_norms - 2 * dots
            max_norms = np.maximum(v_norm_sq, sel_norms)
            
            if not np.all(dist_sqs >= max_norms):
                compatible = False
                
            if compatible:
                selected_indices.append(idx)
                selected_norms_sq.append(v_norm_sq)
                
        if len(selected_indices) > len(best_vectors):
            best_vectors = selected_indices[:]
            
    if not best_vectors:
        # Fallback to seed solution
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return candidates[best_vectors]

