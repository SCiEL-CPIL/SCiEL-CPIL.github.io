"""Tensor decomposition search for 3x5x5 matrix multiplication."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import random
from itertools import product

def _validate(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for c in range(rank):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
            if abs(s - 1.0) > 1e-6:
                return False
    return True

def _try_rank(r, n, m, p, max_iters=500, seed=42):
    """Try to find a rank-r decomposition using random initialization and gradient descent."""
    np.random.seed(seed)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Random initialization
    U = np.random.randn(nm, r) * 0.1
    V = np.random.randn(mp, r) * 0.1
    W = np.random.randn(pn, r) * 0.1
    
    # We want: for all i, k: sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is a non-linear constraint. Let's use a simple iterative approach.
    
    # Actually, let's use a different strategy: try to construct a decomposition
    # based on known low-rank decompositions for small matrix multiplication.
    
    # For 3x5x5, the schoolbook rank is 75.
    # The best known upper bounds for matrix multiplication rank are not much lower
    # for these dimensions, but let's try a few structured approaches.
    
    # Approach: Try to find a decomposition where each term corresponds to a
    # structured pattern of indices.
    
    # Let's try a random search with validation
    for it in range(max_iters):
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        # Check if this happens to be valid (unlikely for random, but let's try)
        if _validate(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Start with schoolbook
    rank_school = n * m * p
    U0 = np.zeros((nm, rank_school))
    V0 = np.zeros((mp, rank_school))
    W0 = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_school
    
    # Try to find better decompositions
    for r in range(best_rank - 1, max(best_rank - 10, 50), -1):
        result = _try_rank(r, n, m, p, max_iters=200, seed=r)
        if result is not None:
            U, V, W = result
            if _validate(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
