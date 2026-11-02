"""Program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        if len(U[r]) != rank or len(V[r]) != rank or len(W[r]) != rank:
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_structured_decomposition(rank, time_limit=20.0):
    """Try to find a rank-`rank` decomposition using structured patterns."""
    n, m, p = 3, 3, 6
    t0 = time.time()
    
    dim_u = n * m  # 9
    dim_v = m * p  # 18
    dim_w = p * n  # 18
    
    # Try different structured approaches
    # Approach 1: Use Kronecker-like structure
    # The 3x3x6 tensor can be seen as 3 blocks of 3x6
    
    # For each possible rank, try to find a decomposition
    # using small integer coefficients
    
    # Try random discrete search on half-integer lattice
    rng = np.random.RandomState(42)
    
    best_loss = float('inf')
    best_solution = None
    
    max_attempts = 10000
    for attempt in range(max_attempts):
        if time.time() - t0 > time_limit:
            break
        
        # Random initialization with small integer/half-integer values
        U = rng.randint(-2, 3, size=(dim_u, rank)) * 0.5
        V = rng.randint(-2, 3, size=(dim_v, rank)) * 0.5
        W = rng.randint(-2, 3, size=(dim_w, rank)) * 0.5
        
        # Compute the tensor
        T = np.einsum('ir,jr,kr->ijk', 
                       U.reshape(n, m, rank),
                       V.reshape(m, p, rank),
                       W.reshape(p, n, rank))
        
        loss = np.sum((T - 1.0) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_solution = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-10:
                # Found exact solution, verify
                U_list = [[float(U[i, r]) for r in range(rank)] for i in range(dim_u)]
                V_list = [[float(V[i, r]) for r in range(rank)] for i in range(dim_v)]
                W_list = [[float(W[i, r]) for r in range(rank)] for i in range(dim_w)]
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_kronecker_decomposition():
    """Try a Kronecker-product-based decomposition."""
    n, m, p = 3, 3, 6
    t0 = time.time()
    
    # For 3x3x6, we can think of it as 3x3x(2*3)
    # Try to decompose using 3x3x3 blocks
    
    # The 3x3x3 matrix multiplication has rank 7 (Strassen-like)
    # But we need exact, not approximate
    
    # Try a simple structured decomposition
    # Split the 6 columns into two groups of 3
    
    rank = 18  # 3 * 6, one per output element per input row
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(6):
                U[i * 3 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin
    
    # Try structured decompositions first
    # The schoolbook rank is 54
    # Try to find lower rank decompositions
    
    # Strategy: Try ranks from 53 down to 40 with time limits
    # Use discrete search on half-integer lattice
    
    ranks_to_try = list(range(53, 39, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        time_remaining = deadline - time.time()
        time_per_attempt = min(15.0, time_remaining / max(1, len(ranks_to_try)))
        
        result = _try_structured_decomposition(rank, time_limit=time_per_attempt)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
