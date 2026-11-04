"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the matmul tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i,j,k
    # Reshape and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_rank(r, n, m, p, max_iters=50, seed=42, deadline=None):
    """Try to find a rank-r decomposition via random initialization and gradient descent.
    Returns (U, V, W) as lists if successful, else None."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, r)
    V = rng.randn(m*p, r)
    W = rng.randn(p*n, r)
    
    lr = 0.01
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return None
        
        # Compute gradients
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # d/dU[i*m+j, c] = V[j*p+k, c] * W[k*n+i, c]
                    # d/dV[j*p+k, c] = U[i*m+j, c] * W[k*n+i, c]
                    # d/dW[k*n+i, c] = U[i*m+j, c] * V[j*p+k, c]
                    for c in range(r):
                        gU[i*m+j, c] += V[j*p+k, c] * W[k*n+i, c]
                        gV[j*p+k, c] += U[i*m+j, c] * W[k*n+i, c]
                        gW[k*n+i, c] += U[i*m+j, c] * V[j*p+k, c]
        
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        # Check convergence
        if it % 10 == 0:
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
                return U.tolist(), V.tolist(), W.tolist()
    
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-9):
        return U.tolist(), V.tolist(), W.tolist()
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, use 115s as safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    best_result = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Try ranks from 21 upward
    min_possible_rank = max(n*m, m*p, p*n)
    
    for r in range(min_possible_rank, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        # Try a few random seeds
        for seed in range(3):
            if time.time() > deadline - 5:
                break
            
            result = _try_rank(r, n, m, p, max_iters=30, seed=seed+r*100, deadline=deadline-5)
            if result is not None:
                U, V, W = result
                # Verify with tight tolerance
                if _verify(U, V, W, n, m, p, tol=1e-9):
                    return {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
