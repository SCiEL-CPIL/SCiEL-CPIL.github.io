"""Matrix multiplication 3x4x5 - rank-minimal decomposition search."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if the given factor matrices decompose the matrix multiplication tensor.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Convention: W[k*n + i, r] corresponds to output C[i, k]
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # D[x, y, z] = sum_r U[x, r] * V[y, r] * W[z, r]
    # x = i*m + j
    # y = j*p + k
    # z = k*n + i
    D = np.einsum('xr,yr,zr->xyz', U_np, V_np, W_np)
    
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                expected[x, y, z] = 1.0
                
    return np.allclose(D, expected, atol=tol)

def _generate_schoolbook(n, m, p):
    """Generate the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _search_lower_rank(n, m, p, max_rank, time_limit):
    """
    Attempt to find a decomposition with rank < max_rank using random search.
    Returns None if no valid decomposition is found within time_limit.
    """
    start_time = time.time()
    current_rank = max_rank - 1
    
    # We try ranks from max_rank-1 downwards.
    # Given the small size, we might just try a few random initializations for ranks close to schoolbook.
    # However, random search for tensor decomposition is hard.
    # We will perform a limited number of random trials for rank = max_rank - 1, etc.
    
    attempts_per_rank = 10
    
    while current_rank >= 1:
        if time.time() - start_time > time_limit:
            break
            
        for _ in range(attempts_per_rank):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization with half-integers (0.5 or 1.0) to satisfy integer/half-integer constraint
            # Note: The problem allows integer/half-integer.
            U = np.random.choice([0, 0.5, 1, -0.5, -1], size=(n * m, current_rank))
            V = np.random.choice([0, 0.5, 1, -0.5, -1], size=(m * p, current_rank))
            W = np.random.choice([0, 0.5, 1, -0.5, -1], size=(p * n, current_rank))
            
            # Simple verification is expensive for large rank, but here rank is small.
            # To speed up, we can check a subset of entries first.
            # But for 3x4x5, full check is fast enough if we vectorize.
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
        current_rank -= 1
        
    return None

def construct():
    """
    Constructs the minimal rank tensor decomposition for 3x4 matrix multiplication.
    Uses a search strategy to attempt lower ranks, falling back to schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a better rank than schoolbook
    # We limit the search time to ensure we don't timeout
    search_time_allowed = 10.0
    
    result = _search_lower_rank(n, m, p, schoolbook_rank, search_time_allowed)
    
    if result is not None:
        return result
        
    # Fallback to schoolbook if search fails or times out
    if time.time() > deadline:
         # Emergency return, though schoolbook generation is instant
         pass
         
    return _generate_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
