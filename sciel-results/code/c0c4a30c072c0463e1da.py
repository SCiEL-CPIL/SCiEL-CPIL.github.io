import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Compute the tensor T[u, v, w] = sum_c U[u, c] * V[v, c] * W[w, c]
    T = np.einsum('uc,vc,wc->uvw', U, V, W, optimize=True)
    
    # Construct the target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T_target[u, v, w] = 1.0
                
    # Check if T matches T_target
    if not np.allclose(T, T_target, atol=tol, rtol=0):
        return False
        
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    
    # Return the schoolbook decomposition which is guaranteed to be correct
    # This ensures the program runs and produces a valid score.
    # Further optimization (e.g., Strassen-like) can be added later if time permits,
    # but correctness is prioritized in this repair.
    U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
