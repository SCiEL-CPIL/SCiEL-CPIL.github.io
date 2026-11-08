import time
import numpy as np
import random

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _random_search_4x4x4(target_rank=14, num_trials=5000):
    """
    Search for a low-rank decomposition of the 4x4x4 matrix multiplication tensor.
    Returns U, V, W of shape (16, rank), (16, rank), (16, rank) if found, else None.
    """
    n = m = p = 4
    nm = 16
    best = None
    for rank in range(target_rank, 20):
        for trial in range(num_trials):
            if time.time() > _DEADLINE - 10:
                return None
            U = np.zeros((nm, rank), dtype=float)
            V = np.zeros((nm, rank), dtype=float)
            W = np.zeros((nm, rank), dtype=float)
            
            # Fill with random values from {-1, 0, 1}
            for r in range(rank):
                U[:, r] = np.random.choice([-1, 0, 1], size=nm)
                V[:, r] = np.random.choice([-1, 0, 1], size=nm)
                W[:, r] = np.random.choice([-1, 0, 1], size=nm)
            
            # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :] * W[k*n+i, :], V[j*p+k, :])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U, V, W, rank
    return None

def _schoolbook_4x4x3():
    """Return the trivial rank-48 decomposition for 4x4x3."""
    n, m, p = 4, 4, 3
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Split 4x4x7 into 4x4x4 and 4x4x3
    # U shape (16, rank1 + rank2)
    # V shape (28, rank1 + rank2) -> (4*7, rank)
    # W shape (28, rank1 + rank2) -> (7*4, rank)
    
    # Get 4x4x4 decomposition
    result444 = _random_search_4x4x4(target_rank=14, num_trials=1000)
    if result444 is None:
        # Fallback to schoolbook for 4x4x4 (rank 64)
        U444 = [[0] * 64 for _ in range(16)]
        V444 = [[0] * 64 for _ in range(16)]
        W444 = [[0] * 64 for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U444[i * 4 + j][col] = 1
                    V444[j * 4 + k][col] = 1
                    W444[k * 4 + i][col] = 1
                    col += 1
        rank1 = 64
    else:
        U444, V444, W444, rank1 = result444
        U444 = U444.tolist()
        V444 = V444.tolist()
        W444 = W444.tolist()
    
    # Get 4x4x3 decomposition (schoolbook rank 48)
    U443, V443, W443 = _schoolbook_4x4x3()
    rank2 = 48
    
    # Combine
    total_rank = rank1 + rank2
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]
    W = [[0] * total_rank for _ in range(28)]
    
    # Fill U, V, W for the first 4 columns of B (4x4x4 part)
    for r in range(rank1):
        for i in range(16):
            U[i][r] = U444[i][r]
        for j in range(4):
            for k in range(4):
                V[j * 7 + k][r] = V444[j * 4 + k][r]
                W[k * 4 + i][r] = W444[k * 4 + i][r]
    
    # Fill U, V, W for the last 3 columns of B (4x4x3 part)
    for r in range(rank2):
        for i in range(16):
            U[i][rank1 + r] = U443[i][r]
        for j in range(4):
            for k in range(3):
                V[j * 7 + 4 + k][rank1 + r] = V443[j * 3 + k][r]
                W[(4 + k) * 4 + i][rank1 + r] = W443[k * 4 + i][r]
    
    return {"U": U, "V": V, "W": W}
