import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    # Evaluator kills at 150s. Leave margin.
    deadline = start_time + 140.0
    
    # We fix c_0 = 1 to reduce search space by half.
    # Search space: 2^23 = 8,388,608 candidates.
    # Coefficients c_1 ... c_23 are free.
    
    # Grid for evaluation.
    # A dense grid is needed to accurately estimate the sup norm.
    # 20,000 to 50,000 points is a good trade-off between speed and accuracy.
    M_grid = 30000
    thetas = np.linspace(0, 2 * np.pi, M_grid, endpoint=False)
    # Precompute powers of z for the grid
    # Z[i, k] = exp(1j * theta_i * k)
    # Shape: (M_grid, N)
    Z = np.exp(1j * np.outer(thetas, np.arange(N)))
    
    # We will search in chunks to keep memory usage reasonable.
    # Chunk size: 100,000 candidates.
    chunk_size = 100000
    total_candidates = 1 << 23  # 2^23
    
    best_score = np.inf
    best_coeffs = None
    
    # Pre-allocate arrays for chunk processing
    # indices: (chunk_size,)
    # bits: (chunk_size, 23)
    # coeffs: (chunk_size, 24)
    
    shifts = np.arange(23)
    
    for start in range(0, total_candidates, chunk_size):
        if time.time() > deadline:
            break
            
        end = min(start + chunk_size, total_candidates)
        count = end - start
        
        # Generate indices for this chunk
        indices = np.arange(start, end, dtype=np.int64)
        
        # Generate bits for c_1 ... c_23
        # bits[i, j] corresponds to coefficient c_{j+1}
        # We want to map integer indices to bit patterns.
        # bits = (indices[:, None] >> shifts[None, :]) & 1
        # Shape: (count, 23)
        bits = (indices[:, None] >> shifts[None, :]) & 1
        
        # Convert bits to +/-1
        # 0 -> 1, 1 -> -1
        c_rest = 1 - 2 * bits
        
        # Prepend c_0 = 1
        # c_0 column: (count, 1)
        c_0 = np.ones((count, 1), dtype=np.int8)
        # coeffs: (count, 24)
        coeffs = np.hstack((c_0, c_rest))
        
        # Evaluate P(z) on the grid
        # coeffs: (count, N)
        # Z.T: (N, M_grid)
        # vals: (count, M_grid)
        vals = coeffs @ Z.T
        
        # Calculate sup norm for each polynomial in the chunk
        # max_abs: (count,)
        max_abs = np.max(np.abs(vals), axis=1)
        
        # Find the minimum in this chunk
        min_idx_in_chunk = np.argmin(max_abs)
        chunk_min_score = max_abs[min_idx_in_chunk]
        
        if chunk_min_score < best_score:
            best_score = chunk_min_score
            best_coeffs = coeffs[min_idx_in_chunk].copy()
            
        # Optional: If we are very close to the known optimum, we might refine.
        # Known optimum is ~1.2583758 * sqrt(24) = ~6.147
        # Our score here is raw max norm.
        # Let's just keep the best raw norm found.
        
    # Refinement:
    # The grid method approximates the sup norm. To be precise, we should refine the peak locations for the best candidate(s).
    # However, for this specific problem, the grid resolution of 30k is usually sufficient to identify the correct polynomial.
    # The "score" in the task description is max|P(z)| / sqrt(24).
    # We return the coefficients.
    
    if best_coeffs is None:
        # Fallback if search failed or timed out immediately
        best_coeffs = np.ones(N, dtype=int)
    else:
        best_coeffs = best_coeffs.astype(int)
        
    # The task asks for 24 coefficients in {+1, -1}
    return [int(x) for x in best_coeffs]
