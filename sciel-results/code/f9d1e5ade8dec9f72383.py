import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 570  # Safety margin for 630s limit
    
    # Candidate pool: all integer vectors with norm^2 <= 4
    # This includes:
    # norm^2=1: 22 vectors (1 at one coord)
    # norm^2=2: 220 vectors (1 at two coords)
    # norm^2=3: 1320 vectors (1 at three coords)
    # norm^2=4: 264 (2 at one coord) + 1320 (1 at four coords) = 1584
    # Total ~ 3146 vectors. This is manageable for vectorized checks.
    
    candidates = []
    
    # Norm 1
    for i in range(d):
        for s in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Norm 2
    for i in range(d):
        for j in range(i+1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
                    
    # Norm 3
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            candidates.append(v)
                            
    # Norm 4
    # Type A: One coordinate is +/-2
    for i in range(d):
        for s in [2, -2]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Type B: Four coordinates are +/-1
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for l in range(k+1, d):
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            for s3 in [1, -1]:
                                for s4 in [1, -1]:
                                    v = np.zeros(d, dtype=np.int64)
                                    v[i] = s1
                                    v[j] = s2
                                    v[k] = s3
                                    v[l] = s4
                                    candidates.append(v)
                                    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates**2, axis=1)
    
    # We will maintain a list of selected indices
    selected_indices = []
    selected_vectors = [] # List of np arrays for easy appending
    
    # Strategy: Greedy with multiple random restarts and different ordering heuristics
    # We want to maximize k.
    # Constraint: For all pairs i, j in selected: ||v_i - v_j||^2 >= max(||v_i||^2, ||v_j||^2)
    # Note: If all have same norm R, then ||v_i - v_j||^2 >= R.
    # Mixing norms is tricky. Usually, keeping norms equal or close is better for packing.
    # Let's try to build a set primarily of one norm, or allow mixed but check strictly.
    
    best_size = 0
    best_set = []
    
    # Helper to check if a new vector v can be added to current set
    # current_set is a list of vectors
    # We need min dist^2 >= max norm^2
    # Let N_max = max(norm^2 of current set, norm^2 of v)
    # For each u in current_set: ||v-u||^2 >= N_max
    
    def can_add(v, current_indices, current_vectors, norm_sq_v):
        if not current_vectors:
            return True
        
        # Max norm is max of existing max and new vector's norm
        # We can track max_norm_sq in the main loop for speed
        # Here we just check against the specific max required
        # Actually, the condition is:
        # For all u in set: ||v-u||^2 >= max( ||v||^2, ||u||^2 )
        
        # Let's compute dists to all current vectors
        # current_vectors is a list of 1-d arrays. 
        # For speed, we might keep a numpy array of selected vectors.
        # But adding one at a time makes it dynamic.
        # Let's use the indices into the global candidates array.
        
        # For now, simple loop check for correctness, optimized later if needed.
        # Since n_cand is ~3000, and we do this ~k times, k~100, it's 300k ops per restart.
        # With vectorization it's fast.
        
        # Let's assume current_vectors is a numpy array of shape (m, 11)
        # This is better for speed.
        
        pass # We will implement the main loop with numpy arrays for selected set.
        
    # Main Loop: Multiple Greedy Trials
    
    # To allow mixed norms, we must be careful.
    # Often, the optimal kissing number in Z^d uses vectors of the same norm.
    # Let's first try to find a large set of a single norm (e.g., norm^2=2 or 3).
    # Then, we can try to add others.
    
    # Strategy A: Greedy MIS on a specific norm pool
    # Strategy B: Dynamic Greedy on full pool
    
    # Let's do Strategy B: Dynamic Greedy
    # We pick a random ordering of candidates.
    # We iterate through them.
    # If compatible with all already selected, add it.
    
    # Optimization: 
    # Keep selected_vectors as a numpy array (m, 11).
    # Keep selected_norms_sq as a numpy array (m,).
    # current_max_norm_sq = max(selected_norms_sq)
    
    for trial in range(200): # Number of restarts
        if time.time() > deadline - 10:
            break
            
        # Shuffle candidates
        perm = np.random.permutation(n_cand)
        
        # Optional: Sort by norm first to encourage uniformity? 
        # Or just random.
        # Let's try random first.
        
        selected_indices = []
        selected_vecs = [] # List for appending, convert to array at end
        selected_norms = []
        current_max_norm = 0
        
        # For speed, we can pre-compute dot products? No, too large (3000x3000).
        # We will compute dists on the fly.
        
        # To speed up, we can skip candidates whose norm is too large relative to current_max_norm?
        # Condition: ||v-u||^2 >= max(||v||^2, ||u||^2)
        # If ||v||^2 is very large, it's harder to satisfy for small u.
        
        for idx in perm:
            if time.time() > deadline - 5:
                break
                
            v = candidates[idx]
            nv_sq = norms_sq[idx]
            
            # Quick check: if selected is empty, add
            if not selected_vecs:
                selected_vecs.append(v)
                selected_norms.append(nv_sq)
                current_max_norm = nv_sq
                continue
                
            # Check compatibility
            # Convert selected_vecs to array for batch operation? 
            # Doing this every step is expensive. 
            # Let's keep a running numpy array? 
            # Appending to numpy array is slow. 
            # List of vectors + periodic conversion? 
            # Or just loop in Python if m is small? 
            # m will grow to ~100-200. 
            # 3000 candidates * 200 trials * 200 checks * 11 dims = 132M ops. 
            # Python loop might be slow. 
            # Let's use numpy for the check.
            
            # We need to check: for all u in S, ||v-u||^2 >= max(nv_sq, ||u||^2)
            
            # Let's convert selected_vecs to array only if len > some threshold? 
            # Or just do it every time? 
            # np.array(list) is O(m). 
            # Let's try to maintain a numpy array `S` and a mask? 
            # No, just converting the list is probably fine for m < 500.
            
            S = np.array(selected_vecs) # (m, 11)
            S_norms = np.array(selected_norms) # (m,)
            
            # Dists squared: ||v-u||^2 = ||v||^2 + ||u||^2 - 2 v.u
            dots = S @ v # (m,)
            dists_sq = nv_sq + S_norms - 2 * dots
            
            # Required min dist for each u:
            # req_i = max(nv_sq, S_norms[i])
            req = np.maximum(nv_sq, S_norms)
            
            if np.all(dists_sq >= req):
                selected_vecs.append(v)
                selected_norms.append(nv_sq)
                if nv_sq > current_max_norm:
                    current_max_norm = nv_sq
                
                # Early stop if we beat best?
                if len(selected_vecs) > best_size:
                    best_size = len(selected_vecs)
                    best_set = selected_vecs[:]
                    # We can't return immediately because we might find better, 
                    # but we can update the record.
                    
        # End candidate loop
        if len(selected_vecs) > best_size:
            best_size = len(selected_vecs)
            best_set = selected_vecs[:]
            
    # After greedy, try local improvement: 
    # Try to add more vectors from the pool that were skipped?
    # Or try to swap? 
    # Given time constraints, let's just verify the best found.
    
    if best_size < 2:
        # Fallback: return a valid small set
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    result = np.array(best_set, dtype=np.int64)
    result = np.around(result).astype(np.int64)
    
    # Final Verification
    max_norm = int(np.max(np.sum(result**2, axis=1)))
    valid = True
    k = len(result)
    for i in range(k):
        for j in range(i + 1, k):
            d2 = int(np.dot(result[i] - result[j], result[i] - result[j]))
            if d2 < max_norm:
                valid = False
                break
        if not valid:
            break
            
    if valid:
        return result
    else:
        # Should not happen if logic is correct, but fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
