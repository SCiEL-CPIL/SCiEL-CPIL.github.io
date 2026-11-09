"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition satisfies T[i,j,k] = 1 for all valid (i,j,k)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _try_rank(r, n, m, p, deadline, seed=42):
    """Try to find a rank-r decomposition using random restarts and local optimization."""
    rng = np.random.RandomState(seed)
    best_err = np.inf
    best_U = best_V = best_W = None
    
    # Precompute indices for vectorized operations
    U_idx = [[i * m + j for j in range(m)] for i in range(n)]
    V_idx = [[j * p + k for k in range(p)] for j in range(m)]
    W_idx = [[k * n + i for i in range(n)] for k in range(p)]
    
    # Indices for slicing
    U_rows = [np.array([i * m + j for j in range(m)]) for i in range(n)]
    V_rows = [np.array([j * p + k for k in range(p)]) for j in range(m)]
    W_rows = [np.array([k * n + i for i in range(n)]) for k in range(p)]
    
    # Indices for W updates (fixed k, varying i)
    W_rows_i = [np.array([k * n + i for i in range(n)]) for k in range(p)]
    
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = rng.randn(n*m, r) * 0.5
        V = rng.randn(m*p, r) * 0.5
        W = rng.randn(p*n, r) * 0.5
        
        for iteration in range(150):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # V part: rows j*p to j*p+p
                    V_part = V[j*p:(j+1)*p, :]
                    # W part: rows k*n+i for k in 0..p-1
                    W_part = W[np.arange(p)*n + i, :]
                    A_mat = V_part * W_part
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # U part: rows i*m+j for i in 0..n-1
                    U_part = U[np.arange(n)*m + j, :]
                    # W part: rows k*n+i for i in 0..n-1
                    W_part = W[k*n + np.arange(n), :]
                    A_mat = U_part * W_part
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # U part: rows i*m+j for j in 0..m-1
                    U_part = U[i*m + np.arange(m), :]
                    # V part: rows j*p+k for j in 0..m-1
                    V_part = V[np.arange(m)*p + k, :]
                    A_mat = U_part * V_part
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (s - 1.0)**2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
    
    if best_err < 1e-8:
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    deadline = time.time() + 145
    
    # Try ranks from 12 up to 23
    for target_rank in range(12, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank(target_rank, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
