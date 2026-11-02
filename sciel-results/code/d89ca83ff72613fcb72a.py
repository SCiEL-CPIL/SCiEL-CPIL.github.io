"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-26 tensor decomposition for 3x3x3 matrix multiplication.
    
    Strategy:
    1. Use Strassen's algorithm (rank 7) for the 2x2 sub-block (indices 0,1).
    2. Use standard rank-1 terms for all elements involving index 2 (the 'border').
    3. Combine these disjoint supports to form a total rank of 7 + 19 = 26.
    
    Returns:
        Dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    R = 26
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, R), dtype=np.int8)
    V = np.zeros((m * p, R), dtype=np.int8)
    W = np.zeros((p * n, R), dtype=np.int8)
    
    # Helper functions for flattened indices
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # --- Part 1: Strassen Decomposition for 2x2 Block (Indices 0,1) ---
    # Format: ( [(u_idx, val), ...], [(v_idx, val), ...], [(w_idx, val), ...] )
    # Correct Strassen terms for 2x2 matrix multiplication:
    # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
    # M2 = (A10 + A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B01 - B11)       -> C01 += M3, C11 += M3
    # M4 = A11(B10 - B00)       -> C00 += M4, C10 += M4
    # M5 = (A00 + A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10 - A00)(B00 + B01) -> C11 += M6
    # M7 = (A01 - A11)(B10 + B11) -> C00 += M7
    
    strassen_terms = [
        # M1
        ([(idx_u(0,0), 1), (idx_u(1,1), 1)],
         [(idx_v(0,0), 1), (idx_v(1,1), 1)],
         [(idx_w(0,0), 1), (idx_w(1,1), 1)]),
         
        # M2
        ([(idx_u(1,0), 1), (idx_u(1,1), 1)],
         [(idx_v(0,0), 1)],
         [(idx_w(1,0), 1), (idx_w(1,1), -1)]),
         
        # M3
        ([(idx_u(0,0), 1)],
         [(idx_v(0,1), 1), (idx_v(1,1), -1)],
         [(idx_w(0,1), 1), (idx_w(1,1), 1)]),
         
        # M4
        ([(idx_u(1,1), 1)],
         [(idx_v(1,0), 1), (idx_v(0,0), -1)],
         [(idx_w(0,0), 1), (idx_w(1,0), 1)]),
         
        # M5
        ([(idx_u(0,0), 1), (idx_u(0,1), 1)],
         [(idx_v(1,1), 1)],
         [(idx_w(0,0), -1), (idx_w(0,1), 1)]),
         
        # M6
        ([(idx_u(1,0), 1), (idx_u(0,0), -1)],
         [(idx_v(0,0), 1), (idx_v(0,1), 1)],
         [(idx_w(1,1), 1)]),
         
        # M7
        ([(idx_u(0,1), 1), (idx_u(1,1), -1)],
         [(idx_v(1,0), 1), (idx_v(1,1), 1)],
         [(idx_w(0,0), 1)]),
    ]
    
    # Populate Strassen terms
    for r, (u_terms, v_terms, w_terms) in enumerate(strassen_terms):
        for idx, val in u_terms:
            U[idx, r] = val
        for idx, val in v_terms:
            V[idx, r] = val
        for idx, val in w_terms:
            W[idx, r] = val
            
    # --- Part 2: Standard Terms for Border (Index 2 involved) ---
    # Total standard multiplications needed: 27 (total) - 8 (handled by 2x2 block) = 19.
    # We iterate over all (i, j, k) where i, j, k in {0,1,2}.
    # If i<2 and j<2 and k<2, skip (handled by Strassen).
    # Else, add a rank-1 term e_{(i,j)} e_{(j,k)} e_{(k,i)}.
    
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                
                # Add rank 1 term
                U[idx_u(i, j), col] = 1
                V[idx_v(j, k), col] = 1
                W[idx_w(k, i), col] = 1
                col += 1
                
    if col != R:
        raise ValueError(f"Rank calculation error: expected {R}, got {col}")
    
    # --- Verification ---
    # The matrix multiplication tensor T is defined by:
    # T[alpha, beta, gamma] = 1 if alpha=(i,j), beta=(j,k), gamma=(i,k)
    # where i, j, k are the logical indices.
    # Otherwise 0.
    
    T_reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Construct the correct target tensor of shape (n*m, m*p, p*n)
    # Note: U index is i*m+j, V index is j*p+k, W index is k*n+i
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.int8)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_target[idx_u, idx_v, idx_w] = 1
    
    if not np.array_equal(T_reconstructed, T_target):
        # Debugging: Find where they differ
        diffs = np.argwhere(T_reconstructed != T_target)
        # This should not happen if the decomposition is correct.
        raise AssertionError(f"Tensor decomposition verification failed. Diffs at {diffs}")
        
    # Check deadline
    if time.time() - start_time > 110:
         # This is a static construction, so we shouldn't be close to the limit,
         # but we return immediately to be safe.
         pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
