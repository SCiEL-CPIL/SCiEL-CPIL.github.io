import numpy as np
import time

DIMENSIONS = [3, 5, 5]
DEADLINE = 140.0

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    # Check non-zero entries (diagonal)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
                
    # Check zero entries (off-diagonal)
    # Sample a few or check all?
    # For small dims, check all.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i == i2 and j == j2 and k == k2:
                                continue
                            val = 0.0
                            for r in range(rank):
                                val += U[i * m + j, r] * V[j2 * p + k2, r] * W[k * n + i2, r]
                            # Note: W index is k*n+i. 
                            # In the tensor T[(i,j),(j2,k2),(i2,k2)], the W index should correspond to the output (i2,k2)?
                            # My W index is k*n+i.
                            # So W[k*n+i] corresponds to output (i,k)?
                            # The term is U[i,j] V[j2,k2] W[?,?].
                            # The output is determined by U and W?
                            # Standard: C_{i,k} = sum_j A_{i,j} B_{j,k}.
                            # Tensor: T_{i,j, j',k, i',k'} = 1 if i=i', j=j', k=k'.
                            # My decomposition: sum_r U[i,j,r] V[j',k,r] W[k,i,r].
                            # This sums to 1 if i=i', j=j', k=k'.
                            # If i != i', the W index is k*n+i.
                            # Does U[i,j] and W[k,i] and V[j',k'] interact?
                            # Yes.
                            if abs(val) > tol:
                                return False
    return True

def _rank1_decomposition(n, m, p):
    rank = 1
    U = np.ones((n * m, rank), dtype=np.float64)
    V = np.ones((m * p, rank), dtype=np.float64)
    W = np.ones((p * n, rank), dtype=np.float64)
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    result = _rank1_decomposition(n, m, p)
    if time.time() - start_time < DEADLINE:
        if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
    
    if time.time() - start_time < DEADLINE:
        return _trivial_decomposition(n, m, p)
        
    return _trivial_decomposition(n, m, p)
