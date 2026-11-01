"""Bootstrap portfolio: MODEL-AND-OPTIMIZE approach for rank-minimal exact decomposition."""
import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T

def _check_decomposition(U, V, W, T):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(U.shape[1]):
        result += np.einsum('u,v,w->uvw', U[:,r].astype(np.float64), 
                            V[:,r].astype(np.float64), W[:,r].astype(np.float64))
    return np.allclose(result, T.astype(np.float64), atol=1e-9)

def _rationalize(U, V, W, max_denom=4):
    """Try to find rational entries with denominator <= max_denom that satisfy the equations."""
    def snap_matrix(M):
        return np.round(M * max_denom) / max_denom
    
    U_s = snap_matrix(U)
    V_s = snap_matrix(V)
    W_s = snap_matrix(W)
    return U_s, V_s, W_s

def _als_one_factor(T, V, W, n, m, p, R):
    """Solve for U given V, W using least squares."""
    U_rows = n * m
    VvW = np.einsum('vr,wr->vwr', V, W)  # shape (m*p, R, p*n)
    VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)  # shape (m*p * p*n, R)
    
    U = np.zeros((U_rows, R), dtype=np.float64)
    for u in range(U_rows):
        b = T[u].reshape(-1).astype(np.float64)
        sol, residuals, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
        U[u] = sol
    return U

def _als_one_factor_V(T, U, W, n, m, p, R):
    """Solve for V given U, W."""
    V_rows = m * p
    Uw = np.einsum('ur,wr->uwr', U, W)  # shape (n*m, p*n, R)
    Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)  # shape (n*m * p*n, R)
    
    V = np.zeros((V_rows, R), dtype=np.float64)
    for v in range(V_rows):
        T_v = T[:, v, :]  # shape (n*m, p*n)
        b = T_v.reshape(-1).astype(np.float64)
        sol, residuals, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
        V[v] = sol
    return V

def _als_one_factor_W(T, U, V, n, m, p, R):
    """Solve for W given U, V."""
    W_rows = p * n
    UV = np.einsum('ur,vr->uvr', U, V)  # shape (n*m, m*p, R)
    UV_flat = UV.reshape(-1, R)  # shape (n*m * m*p, R)
    
    W = np.zeros((W_rows, R), dtype=np.float64)
    for w in range(W_rows):
        T_w = T[:, :, w]  # shape (n*m, m*p)
        b = T_w.reshape(-1).astype(np.float64)
        sol, residuals, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W

def _try_rank_als(R, n, m, p, T, max_iters=300, seed=42, init_scale=0.5):
    """Try to find a rank-R decomposition using ALS."""
    np.random.seed(seed)
    U = np.random.randn(n*m, R) * init_scale
    V = np.random.randn(m*p, R) * init_scale
    W = np.random.randn(p*n, R) * init_scale
    
    prev_error = np.inf
    for it in range(max_iters):
        U = _als_one_factor(T, V, W, n, m, p, R)
        V = _als_one_factor_V(T, U, W, n, m, p, R)
        W = _als_one_factor_W(T, U, V, n, m, p, R)
        
        # Compute error
        err = 0.0
        for r in range(R):
            err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T.astype(np.float64))**2)
        err = np.sqrt(err / (n*m * m*p * p*n))
        
        if abs(prev_error - err) < 1e-14:
            break
        prev_error = err
    
    if _check_decomposition(U, V, W, T):
        for denom in [2, 4, 8]:
            U_s, V_s, W_s = _rationalize(U, V, W, max_denom=denom)
            if _check_decomposition(U_s, V_s, W_s, T):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def _try_rank_structured(R, n, m, p, T, seed=42):
    """Try structured initializations based on schoolbook with perturbations."""
    if R < 1:
        return None
    
    np.random.seed(seed)
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    
    # Strategy: take first R columns of schoolbook and try to adjust
    if R <= n * m * p:
        U_init = U_sb[:, :R].copy()
        V_init = V_sb[:, :R].copy()
        W_init = W_sb[:, :R].copy()
        
        # Add small random perturbation to break symmetry
        U_init += np.random.randn(n*m, R) * 0.01
        V_init += np.random.randn(m*p, R) * 0.01
        W_init += np.random.randn(p*n, R) * 0.01
        
        # Run ALS from this initialization
        prev_error = np.inf
        for it in range(200):
            U_init = _als_one_factor(T, V_init, W_init, n, m, p, R)
            V_init = _als_one_factor_V(T, U_init, W_init, n, m, p, R)
            W_init = _als_one_factor_W(T, U_init, V_init, n, m, p, R)
            
            err = 0.0
            for r in range(R):
                err += np.sum((np.einsum('u,v,w->uvw', U_init[:,r], V_init[:,r], W_init[:,r]) - T.astype(np.float64))**2)
            err = np.sqrt(err / (n*m * m*p * p*n))
            
            if abs(prev_error - err) < 1e-14:
                break
            prev_error = err
        
        if _check_decomposition(U_init, V_init, W_init, T):
            for denom in [2, 4, 8]:
                U_s, V_s, W_s = _rationalize(U_init, V_init, W_init, max_denom=denom)
                if _check_decomposition(U_s, V_s, W_s, T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
            return U_init.tolist(), V_init.tolist(), W_init.tolist()
    
    return None

def construct():
    deadline = time.time() + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    T = _build_target()
    
    # First try structured approaches for various ranks
    for R in range(14, 21):
        if time.time() > deadline:
            break
        for seed in [42, 123, 7, 999, 2024, 555, 777, 1234]:
            if time.time() > deadline:
                break
            try:
                result = _try_rank_structured(R, n, m, p, T, seed=seed)
                if result is not None:
                    U_l, V_l, W_l = result
                    return {"U": U_l, "V": V_l, "W": W_l}
            except Exception:
                continue
    
    # Then try random ALS for lower ranks
    for R in range(12, 21):
        if time.time() > deadline:
            break
        for seed in [42, 123, 7, 999, 2024, 555, 777, 1234, 3141, 2718]:
            if time.time() > deadline:
                break
            try:
                result = _try_rank_als(R, n, m, p, T, max_iters=200, seed=seed, init_scale=0.3)
                if result is not None:
                    U_l, V_l, W_l = result
                    return {"U": U_l, "V": V_l, "W": W_l}
            except Exception:
                continue
    
    # Fall back to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
