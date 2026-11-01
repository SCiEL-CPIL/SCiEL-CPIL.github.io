"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations
from scipy.optimize import nnls
import random

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    result = np.zeros_like(T)
    for r in range(U.shape[1]):
        result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _rationalize(U, V, W, max_denom=2):
    """Snap to nearest 1/max_denom and verify."""
    def snap_matrix(M):
        return np.round(M * max_denom) / max_denom
    U_s = snap_matrix(U)
    V_s = snap_matrix(V)
    W_s = snap_matrix(W)
    return U_s, V_s, W_s

def _als_solve_U(T, V, W, R):
    """Solve for U given V, W."""
    n, m, p = DIMENSIONS
    VvW = np.einsum('vr,wr->vwr', V, W)
    VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
    U = np.zeros((n*m, R), dtype=np.float64)
    for u in range(n*m):
        b = T[u].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(VvW_flat, b, rcond=None)
        U[u] = sol
    return U

def _als_solve_V(T, U, W, R):
    """Solve for V given U, W."""
    n, m, p = DIMENSIONS
    Uw = np.einsum('ur,wr->uwr', U, W)
    Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
    V = np.zeros((m*p, R), dtype=np.float64)
    for v in range(m*p):
        T_v = T[:, v, :]
        b = T_v.reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(Uw_flat, b, rcond=None)
        V[v] = sol
    return V

def _als_solve_W(T, U, V, R):
    """Solve for W given U, V."""
    n, m, p = DIMENSIONS
    UV = np.einsum('ur,vr->uvr', U, V)
    UV_flat = UV.reshape(-1, R)
    W = np.zeros((p*n, R), dtype=np.float64)
    for w in range(p*n):
        T_w = T[:, :, w]
        b = T_w.reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W

def _try_rank_als(R, n, m, p, T, max_iters=150, seed=42):
    """Try to find a rank-R decomposition using ALS."""
    np.random.seed(seed)
    U = np.random.randn(n*m, R) * 0.5
    V = np.random.randn(m*p, R) * 0.5
    W = np.random.randn(p*n, R) * 0.5
    
    prev_error = np.inf
    for it in range(max_iters):
        U = _als_solve_U(T, V, W, R)
        V = _als_solve_V(T, U, W, R)
        W = _als_solve_W(T, U, V, R)
        
        # Compute error
        result = np.zeros_like(T)
        for r in range(R):
            result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
        err = np.sqrt(np.sum((result - T)**2) / (n*m * m*p * p*n))
        
        if abs(prev_error - err) < 1e-13:
            break
        prev_error = err
    
    if _check_decomposition(U, V, W, T):
        for denom in [1, 2, 4]:
            U_s, V_s, W_s = _rationalize(U, V, W, max_denom=denom)
            if _check_decomposition(U_s, V_s, W_s, T):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    T = _build_target()
    
    # Try ranks from 14 to 19 (below schoolbook 20)
    # Use multiple seeds for each rank
    seeds = [42, 123, 7, 999, 2024, 555, 777, 12345, 67890, 31415]
    
    for R in range(14, 20):
        if time.time() > deadline:
            break
        for seed in seeds:
            if time.time() > deadline:
                break
            try:
                result = _try_rank_als(R, n, m, p, T, max_iters=100, seed=seed)
                if result is not None:
                    U_l, V_l, W_l = result
                    # Final verification
                    U_arr = np.array(U_l, dtype=float)
                    V_arr = np.array(V_l, dtype=float)
                    W_arr = np.array(W_l, dtype=float)
                    if _check_decomposition(U_arr, V_arr, W_arr, T):
                        return {"U": U_l, "V": V_l, "W": W_l}
            except Exception:
                continue
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
