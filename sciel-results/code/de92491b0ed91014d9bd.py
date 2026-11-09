import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 110
    
    powers = np.arange(N)
    
    # Coarse grid for filtering
    # Using 1024 points is a good balance between speed and accuracy for initial filtering
    M_coarse = 1024
    thetas_coarse = np.linspace(0, 2 * np.pi, M_coarse, endpoint=False)
    Z_coarse = np.exp(1j * thetas_coarse[:, None] * powers[None, :]).astype(np.float32)
    
    # Fine grid for high-precision refinement
    # Using a dense grid (20000 points) is faster and more reliable than golden section search in Python
    M_fine = 20000
    thetas_fine = np.linspace(0, 2 * np.pi, M_fine, endpoint=False)
    Z_fine = np.exp(1j * thetas_fine[:, None] * powers[None, :])
    
    # Search space: fix c0=1, c1=1.
    # Free bits: c2...c23 (22 bits)
    total_combinations = 1 << 22
    chunk_size = 50000
    
    # List to hold (score, coeffs)
    all_candidates = []
    
    # Helper function to evaluate sup norm using dense grid
    def evaluate_dense(c):
        # c is a numpy array of shape (N,) with values +/- 1
        # Evaluate on fine grid
        vals = Z_fine @ c
        abs_vals = np.abs(vals)
        return np.max(abs_vals)

    for start in range(0, total_combinations, chunk_size):
        if time.time() > deadline - 20: # Leave 20s for refinement
            break
            
        end = min(start + chunk_size, total_combinations)
        count = end - start
        
        # Generate indices
        indices = np.arange(start, end)
        
        # Generate bits for c2...c23
        shifts = np.arange(22)
        bits = (indices[:, None] >> shifts[None, :]) & 1
        
        # Convert to +1/-1
        c_rest = 1 - 2 * bits
        
        # Prepend c0=1, c1=1
        c01 = np.ones((count, 2), dtype=np.float32)
        coeffs = np.hstack((c01, c_rest.astype(np.float32)))
        
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
            
    # Sort all_candidates by coarse score (ascending)
    all_candidates.sort(key=lambda x: x[0])
    
    # Take top 100 for refinement
    top_refine = all_candidates[:100]
    
    best_c = None
    best_score = np.inf
    
    for score, c in top_refine:
        if time.time() > deadline - 5:
            break
            
        # Evaluate with dense grid refinement
        fine_score = evaluate_dense(c)
        
        if fine_score < best_score:
            best_score = fine_score
            best_c = c.copy()
            
    # Fallback if no candidates or error
    if best_c is None:
        best_c = np.ones(N, dtype=int)
    else:
        best_c = best_c.astype(int)
        
    return [int(x) for x in best_c]
