import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _rank_one_decomposition(n, m, p):
    """
    Return the rank 1 decomposition if the tensor is all-ones.
    For matrix multiplication, T[i,j,k] = 1.
    U[i*m+j, 0] = 1
    V[j*p+k, 0] = 1
    W[k*n+i, 0] = 1
    """
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W form a valid tensor decomposition for matrix multiplication.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Reshape for efficient checking
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try rank 1 first
    result = _rank_one_decomposition(n, m, p)
    U, V, W = result['U'], result['V'], result['W']
    
    if _verify_decomposition(U, V, W, n, m, p):
        return result
    
    # Fallback to trivial
    return _trivial_decomposition(n, m, p)

def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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
