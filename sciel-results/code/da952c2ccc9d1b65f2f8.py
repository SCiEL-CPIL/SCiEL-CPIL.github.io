import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 110
    
    # Coarse grid for filtering
    M_coarse = 1024
    thetas_coarse = np.linspace(0, 2 * np.pi, M_coarse, endpoint=False)
    Z_coarse = np.exp(1j * thetas_coarse[:, None] * np.arange(N)[None, :])
    
    # Fine grid for final evaluation
    M_fine = 20000
    thetas_fine = np.linspace(0, 2 * np.pi, M_fine, endpoint=False)
    Z_fine = np.exp(1j * thetas_fine[:, None] * np.arange(N)[None, :])
    
    # Search space: fix c0=1, c1=1.
    # Free bits: c2...c23 (22 bits)
    total_combinations = 1 << 22
    chunk_size = 50000
    
    # List to hold (score, coeffs)
    # We store (coarse_score, coeffs)
    all_candidates = []
    
    for start in range(0, total_combinations, chunk_size):
        if time.time() > deadline - 30: # Leave 30s for refinement
            break
            
        end = min(start + chunk_size, total_combinations)
        count = end - start
        
        # Generate indices
        indices = np.arange(start, end)
        
        # Generate bits for c2...c23
        # bits shape (count, 22)
        shifts = np.arange(22)
        bits = (indices[:, None] >> shifts[None, :]) & 1
        
        # Convert to +1/-1
        c_rest = 1 - 2 * bits
        
        # Prepend c0=1, c1=1
        c01 = np.ones((count, 2))
        coeffs = np.hstack((c01, c_rest))
        
        # Evaluate on coarse grid
        vals = coeffs @ Z_coarse.T
        max_norms = np.max(np.abs(vals), axis=1)
        
        # Find top K (smallest norms) in this chunk
        K = 50
        if count > K:
            part_idx = np.argpartition(max_norms, K)[:K]
            top_scores = max_norms[part_idx]
            top_coeffs = coeffs[part_idx]
        else:
            top_scores = max_norms
            top_coeffs = coeffs
            
        for i in range(len(top_scores)):
            all_candidates.append((top_scores[i], top_coeffs[i]))
            
    # If we ran out of time, we might have few candidates.
    # Sort all_candidates by coarse score (ascending)
    all_candidates.sort(key=lambda x: x[0])
    
    # Take top 100 for refinement
    top_refine = all_candidates[:100]
    
    best_c = None
    best_score = np.inf
    
    for score, c in top_refine:
        if time.time() > deadline - 5:
            break
        # Evaluate with fine grid
        vals_fine = Z_fine @ c
        fine_score = np.max(np.abs(vals_fine))
        
        if fine_score < best_score:
            best_score = fine_score
            best_c = c.copy()
            
    # Fallback if no candidates or error
    if best_c is None:
        best_c = np.ones(N, dtype=int)
    else:
        best_c = best_c.astype(int)
        
    return [int(x) for x in best_c]
