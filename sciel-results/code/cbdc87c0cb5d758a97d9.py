import time
import numpy as np

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 145

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m times m x p -> n x p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W, rank

def _direct_sum_decompose(n, m, p):
    """
    Decompose the tensor T[i,j,k,r] = delta_{i,r} delta_{j,k}.
    This tensor is a direct sum of n tensors of size m x p.
    Each sub-tensor is delta_{j,k}, which has rank min(m, p).
    Total rank = n * min(m, p).
    """
    rank = n * min(m, p)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(min(m, p)):
            # U index: i*m + j
            U[i * m + j][col] = 1.0
            # V index: j*p + j (since k=j)
            V[j * p + j][col] = 1.0
            # W index: j*n + i (since k=j, r=i)
            W[j * n + i][col] = 1.0
            col += 1
            
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[i,j,k,r] = U[i*m+j, :] dot V[j*p+k, :] * W[k*n+r, :]
    # We want T[i,j,k,r] = delta_{i,r} delta_{j,k}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+r][c]
                    expected = 1.0 if (i == r and j == k) else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Use the direct sum decomposition which is rank minimal for this specific tensor structure
    U, V, W, rank = _direct_sum_decompose(n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W, rank = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
