"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_rank(R, n, m, p, timeout=5, seed=42):
    """Try to find a rank-R decomposition using alternating least squares."""
    start = time.time()
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, R) * 0.5
    V = rng.randn(m*p, R) * 0.5
    W = rng.randn(p*n, R) * 0.5
    for it in range(500):
        if time.time() - start > timeout:
            break
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, R))
                for j in range(m):
                    for r in range(R):
                        A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                try:
                    x, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        # Update U
        for i in range(n):
            for j in range(m):
                B = np.zeros((p, R))
                for k in range(p):
                    for r in range(R):
                        B[k, r] = V[j*p+k, r] * W[k*n+i, r]
                try:
                    x, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        # Update V
        for j in range(m):
            for k in range(p):
                C = np.zeros((n, R))
                for i in range(n):
                    for r in range(R):
                        C[i, r] = U[i*m+j, r] * W[k*n+i, r]
                try:
                    x, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
    err = _verify(U, V, W, n, m, p)
    if err < 1e-10:
        return U, V, W, err
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    deadline = time.time() + 110
    for R in range(12, rank_school + 1):
        if time.time() > deadline:
            break
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            result = _try_rank(R, n, m, p, timeout=3, seed=seed)
            if result is not None:
                U, V, W, err = result
                best_U, best_V, best_W = U, V, W
                best_rank = R
                found = True
                break
        if found:
            break
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
