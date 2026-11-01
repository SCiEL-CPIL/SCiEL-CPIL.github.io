"""Program for matrix multiplication 2x5x5 tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Trivial decomposition with rank = n*m*p = 50
    # But we can try to do better with a structured decomposition.
    
    # For a 2x5x5 matrix multiplication, the trivial rank is 50.
    # The best known upper bounds are around 38-40 using Strassen-like methods.
    # However, implementing a verified rank-38 decomposition is complex.
    # Let's use the trivial decomposition which is guaranteed correct.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i*m + j
                # B index: j*p + k
                # C index: k*n + i (trace convention)
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
