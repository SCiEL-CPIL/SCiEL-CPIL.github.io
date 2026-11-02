import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = time.time() + 110

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify decomposition.
    U: list of lists (n*m, rank)
    V: list of lists (m*p, rank)
    W: list of lists (p*n, rank)
    """
    rank = len(U[0]) if n*m > 0 else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    # U indices: i*m + j -> (i, j)
    # V indices: j*p + k -> (j, k)
    # W indices: k*n + i -> (k, i)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Using einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=tol)

def _schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    n, m, p = DIMENSIONS
    result = _schoolbook()
    
    # Verify to be safe
    if _verify(result['U'], result['V'], result['W'], n, m, p):
        return result
    else:
        # This should never happen for schoolbook, but return anyway
        return result
