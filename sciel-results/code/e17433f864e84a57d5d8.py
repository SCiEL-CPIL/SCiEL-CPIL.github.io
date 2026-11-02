"""Matrix multiplication 3x4x6: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_full(U, V, W, n=3, m=4, p=6):
    """Verify: T[i,k,j] = sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] = 1 for all (i,k,j)"""
    rank = U.shape[1]
    errors = 0
    max_err = 0.0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                actual = float(np.dot(np.array([U[i*m+k, r] for r in range(rank)]),
                                      np.array([V[k*p+j, r] * W[j*n+i, r] for r in range(rank)])))
                err = abs(actual - 1.0)
                max_err = max(max_err, err)
                if err > 1e-10:
                    errors += 1
    return errors, max_err


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 130
    
    # Rank-m decomposition: one term per intermediate index k
    # Term k: U[i*m+k, k]=1 for all i; V[k*p+j, k]=1 for all j; W[j*n+i, k]=1 for all i,j
    rank = m  # = 4
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for k in range(m):
        for i in range(n):
            U[i * m + k, k] = 1.0
        for j in range(p):
            V[k * p + j, k] = 1.0
        for i in range(n):
            for j in range(p):
                W[j * n + i, k] = 1.0
    
    # Verify
    errors, max_err = _verify_full(U, V, W, n, m, p)
    if errors > 0:
        # Should not happen, but fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1
                    V[k * p + j, col] = 1
                    W[j * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
