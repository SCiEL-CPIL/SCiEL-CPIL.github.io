"""Tensor decomposition search for 4x4 * 4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _build_tensors(n, m, p):
    """Build the standard basis matrices for the tensor T_{i,j,k} = e_i^T A e_j e_k^T B e_l -> C_{i,l}.
    Here we use the convention:
      U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
      and the tensor is T = sum_c U[:,c] V[:,c]^T W[:,c]^T
    The tensor T has entries T[(i,j),(j,k),(k,i)] = 1 for all i,j,k.
    We reshape T to a 3D array T3[i, j, k] = 1 for all i,j,k.
    Then we flatten to get a 2D matrix T2 of shape (n*m, m*p) with entries
    T2[i*m+j, j*p+k] = 1 if the (i,j) and (j,k) share the j index.
    Actually, the tensor for matrix multiplication is:
      C_{i,k} = sum_j A_{i,j} B_{j,k}
    The tensor T has indices (i,j) for A, (j,k) for B, and (i,k) for C.
    T_{(i,j),(j,k),(i,k)} = 1.
    
    So U has rows indexed by (i,j), V has rows indexed by (j,k), W has rows indexed by (i,k).
    T3[i, j, k] = 1 for all i, j, k.
    """
    # T3[i, j, k] = 1 for all i, j, k
    T3 = np.ones((n, m, p), dtype=np.float64)
    return T3

def _decompose_tensor(T3, n, m, p, max_rank, time_limit=100):
    """Try to find a decomposition of rank at most max_rank.
    T3[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    start = time.time()
    best = None
    
    # Start from schoolbook rank
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    best = (rank, U, V, W)
    
    # Try to reduce rank by checking if some columns are linearly dependent
    # This is a simple approach; more sophisticated methods would be needed for real improvement
    
    # Check if we can reduce rank by 1
    for new_rank in range(rank - 1, max(0, rank - 10), -1):
        if time.time() - start > time_limit:
            break
        # Try to find a decomposition of rank new_rank
        # This is a hard problem; we'll just return the best we have
        break
    
    return best

def construct():
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
# EVOLVE-BLOCK-END
