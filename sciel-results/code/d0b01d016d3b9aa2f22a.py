import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Generate Candidate Vectors
    # We will consider vectors with norm squared <= 4 (Norm <= 2)
    # This covers norms 1, sqrt(2), sqrt(3), 2.
    
    candidates = []
    
    # Norm 1: e_i, -e_i
    for i in range(D):
        v = [0]*D
        v[i] = 1
        candidates.append(v)
        v[i] = -1
        candidates.append(v)
        
    # Norm sqrt(2): e_i +/- e_j
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*D
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Norm sqrt(3): e_i +/- e_j +/- e_k
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
                    
    # Norm 2: 
    # Type A: 2*e_i
    for i in range(D):
        v = [0]*D
        v[i] = 2
        candidates.append(v)
        v[i] = -2
        candidates.append(v)
        
    # Type B: e_i +/- e_j +/- e_k +/- e_l (4 non-zeros)
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0]*D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates.append(v)
                        
    # Convert to numpy array
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # We will test different R values.
    # R is the max norm allowed.
    # The constraint is min pairwise dist >= R.
    
    best_vectors = None
    best_score = 0
    
    # Possible R values based on candidate norms
    # Norms squared: 1, 2, 3, 4
    # R values: 1, sqrt(2), sqrt(3), 2
    
    # We prioritize larger R if it allows more vectors?
    # Usually, smaller R allows denser packing, but limits the "size" of vectors.
    # However, the score is just the count k.
    # So we want to maximize k.
    
    # Let's try R = sqrt(2) first (Seed case)
    # Then R = sqrt(3)
    # Then R = 2
    
    # Actually, let's just run a search for each R.
    
    for r_sq in [2, 3, 4]: # Norm squared thresholds
        if time.time() > deadline - 20: break
        
        # Filter candidates with norm_sq <= r_sq
        norms_sq = np.sum(candidates**2, axis=1)
        mask = norms_sq <= r_sq
        active_candidates = candidates[mask]
        n_active = len(active_candidates)
        
        if n_active == 0: continue
        
        # Compute conflict matrix
        # Conflict if dist_sq < r_sq
        # dist_sq(u,v) = ||u||^2 + ||v||^2 - 2 u.v
        
        # To save memory and time, we can process in chunks or use sparse representation?
        # n_active is at most ~6800.
        # 6800x6800 matrix is ~46M entries.
        # If we store as bool, it's ~5.7 MB.
        
        # Calculate pairwise squared distances
        # Using broadcasting: (n, 1, D) - (1, n, D)
        # This creates an (n, n, D) array.
        # 6800 * 6800 * 11 * 8 bytes = ~4 GB. Too much memory.
        
        # We must compute in chunks or use a more efficient method.
        # Or simply loop? 6800^2 / 2 = 23M iterations.
        # Python loop is too slow.
        # NumPy chunking:
        
        conflict_matrix = np.zeros((n_active, n_active), dtype=bool)
        chunk_size = 500
        
        norms_sq_active = np.sum(active_candidates**2, axis=1)
        
        for i_start in range(0, n_active, chunk_size):
            i_end = min(i_start + chunk_size, n_active)
            if time.time() > deadline - 10: break
            
            # active_candidates[i_start:i_end] shape (chunk, D)
            # active_candidates[:, :j] ...
            
            # Compute distances from chunk to all
            # (chunk, 1, D) - (1, n, D) -> (chunk, n, D)
            # Sum squares -> (chunk, n)
            
            diff = active_candidates[i_start:i_end, None, :] - active_candidates[None, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            
            # Conflict if dist_sq < r_sq
            # Note: dist_sq[i,i] = 0 < r_sq, so diagonal is True.
            # We don't want self-conflict, but it doesn't matter for MIS if we handle it.
            
            conflict_matrix[i_start:i_end, :] = dist_sq < r_sq
            
        # Now find Maximum Independent Set in conflict_matrix
        # Greedy + Local Search
        
        # Initialize
        selected = np.zeros(n_active, dtype=bool)
        
        # Greedy Pass
        # Random order
        order = np.random.permutation(n_active)
        
        for idx in order:
            if time.time() > deadline - 5: break
            # Check if idx conflicts with any selected
            # conflict_matrix[idx, selected]
            if not np.any(conflict_matrix[idx, selected]):
                selected[idx] = True
                
        current_size = np.sum(selected)
        
        # Local Search: Try to swap
        # Simple improvement:
        # Try to add unselected nodes that don't conflict? (Greedy already did this)
        # Try to remove one selected node to allow adding two unselected?
        
        # Given time, a simple restart strategy is better.
        # Run multiple random restarts of the greedy phase.
        
        best_local_size = current_size
        best_local_selected = selected.copy()
        
        restarts = 0
        while time.time() < deadline - 10 and restarts < 5:
            restarts += 1
            selected = np.zeros(n_active, dtype=bool)
            order = np.random.permutation(n_active)
            
            for idx in order:
                if not np.any(conflict_matrix[idx, selected]):
                    selected[idx] = True
                    
            current_size = np.sum(selected)
            if current_size > best_local_size:
                best_local_size = current_size
                best_local_selected = selected.copy()
                
        if best_local_size > best_score:
            best_score = best_local_size
            best_vectors = active_candidates[best_local_selected]
            
    # Fallback to seed if search failed or timed out
    if best_vectors is None or len(best_vectors) == 0:
         vectors = []
         for i, j in itertools.combinations(range(D), 2):
             for si in (1, -1):
                 for sj in (1, -1):
                     v = [0] * D
                     v[i], v[j] = si, sj
                     vectors.append(v)
         best_vectors = np.array(vectors, dtype=np.int64)
         
    return best_vectors
# EVOLVE-BLOCK-END
