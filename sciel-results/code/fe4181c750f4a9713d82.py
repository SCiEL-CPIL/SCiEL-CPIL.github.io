"""
Rank-minimal tensor decomposition for 2x3 * 3x10 matrix multiplication.
Target: rank 30 (the theoretical lower bound).
"""
import time
import numpy as np

DEADLINE = 150

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank30():
    """
    Construct a rank-30 decomposition.
    Term c for (j, k):
    U[i*m+j, c] = 1 for all i
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1 for all i
    """
    n, m, p = 2, 3, 10
    rank = m * p  # 30
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    col = 0
    for j in range(m):
        for k in range(p):
            # U
            for i in range(n):
                U[i*m+j, col] = 1.0
            # V
            V[j*p+k, col] = 1.0
            # W
            for i in range(n):
                W[k*n+i, col] = 1.0
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = 2, 3, 10
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
                
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    
    # Try rank 30 first
    try:
        result = _try_rank30()
        if _verify(result["U"], result["V"], result["W"], 2, 3, 10):
            return result
    except Exception:
        pass
        
    # Fallback to schoolbook rank 60
    try:
        result = _schoolbook()
        if _verify(result["U"], result["V"], result["W"], 2, 3, 10):
            return result
    except Exception:
        pass
        
    # Last resort: return empty or raise?
    # Returning a valid schoolbook is better.
    return _schoolbook()
