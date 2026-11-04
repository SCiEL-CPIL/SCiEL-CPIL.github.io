import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_49(n, m, p):
    """
    Attempt to find rank-49 decomposition for 2x5x5 matrix multiplication.
    Uses HOOI-like iterative refinement initialized from random factors,
    then rounds to rationals.
    """
    if time.time() > _DEADLINE - 20:
        return None

    target_rank = 49
    # Dimensions: n=2, m=5, p=5
    # U: (10, 49), V: (25, 49), W: (10, 49)
    
    try:
        # Precompute valid indices
        valid_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    valid_indices.append((u_idx, v_idx, w_idx))
        
        max_restarts = 8
        for restart in range(max_restarts):
            if time.time() > _DEADLINE - 5:
                break
            
            np.random.seed(100 + restart)
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # HOOI-like iterative refinement
            for iteration in range(200):
                if time.time() > _DEADLINE - 5:
                    break
                
                # Update U: for each (i,j) row, solve least squares
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        # Collect all k values
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            v_idx = j * p + k
                            w_idx = k * n + i
                            A[k, :] = V[v_idx, :] * W[w_idx, :]
                        # Solve A @ u_vec = 1
                        try:
                            U[u_idx, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k) row
                for j in range(m):
                    for k in range(p):
                        v_idx = j * p + k
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            u_idx = i * m + j
                            w_idx = k * n + i
                            A[i, :] = U[u_idx, :] * W[w_idx, :]
                        try:
                            V[v_idx, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i) row
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            A[j, :] = U[u_idx, :] * V[v_idx, :]
                        try:
                            W[w_idx, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                error = 0.0
                for u_idx, v_idx, w_idx in valid_indices:
                    s = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    error += (s - 1.0)**2
                if error < 1e-10:
                    break
            
            if error < 1e-6:
                # Try rounding to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    
                # Try rounding to nearest integer
                U_r = np.round(U)
                V_r = np.round(V)
                W_r = np.round(W)
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # Try rounding to nearest quarter
                U_r = np.round(U * 4) / 4
                V_r = np.round(V * 4) / 4
                W_r = np.round(W * 4) / 4
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
    except Exception:
        pass
        
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find rank 49
    result = _try_rank_49(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
