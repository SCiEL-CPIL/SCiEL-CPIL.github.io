"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 if j is the "middle" index
    # The matrix multiplication tensor T[i,j,k] = 1 if the indices align properly
    # For matrix multiplication C[i,k] = sum_j A[i,j] B[j,k], the tensor has T[i,j,k] = 1 for all valid i,j,k
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_rank(r, n, m, p, max_iters=1000, timeout=10):
    """Try to find a rank-r decomposition using random initialization and verification."""
    start_time = time.time()
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        U = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(n * m)]
        V = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(m * p)]
        W = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(p * n)]
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin for 150s limit
    
    # Try to find lower-rank decompositions
    # Start from rank 99 and go down, but be smart about it
    # For 4x5x5, let's try a few ranks
    best = _schoolbook(n, m, p)
    
    # Try ranks from 95 down to 80, with limited time each
    for r in range(95, 79, -1):
        if time.time() > deadline:
            break
        result = _try_rank(r, n, m, p, max_iters=500, timeout=5)
        if result is not None:
            best = result
            break
    
    return best
# EVOLVE-BLOCK-END
