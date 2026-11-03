import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Schoolbook decomposition: rank = n * m * p = 100
    # This is always valid and guaranteed to work.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 if C[i,k] gets contribution from A[i,j]*B[j,k]
    
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Indexing: U[(i*m + j), r], V[(j*p + k), r], W[(k*n + i), r]
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
