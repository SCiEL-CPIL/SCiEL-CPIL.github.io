import time
import numpy as np
from scipy.optimize import least_squares

_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 150s kill

def _time_remaining():
    return _DEADLINE - time.time()

def _get_target_tensor():
    n, m, p = 2, 2, 7
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _verify(U, V, W, T):
    T_rec = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_rec += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.allclose(T_rec, T, atol=1e-9)

def _is_half_integer(X, tol=1e-6):
    return np.allclose(X * 2, np.round(X * 2), atol=tol)

def _construct_rank_25():
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start+0] += 1; U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1; V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1; W[2*(k_start+1) + 1, col_start+0] += 1
        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1; U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1; W[2*(k_start+1) + 1, col_start+1] -= 1
        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1; V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1; W[2*(k_start+1) + 1, col_start+2] += 1
        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1; V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1; W[2*k_start + 1, col_start+3] += 1
        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start+4] += 1; U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1; W[2*(k_start+1) + 0, col_start+4] += 1
        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start+5] += 1; U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1; V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1
        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start+6] += 1; U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1; V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
    
    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    col_start = 21
    U[0, col_start+0] += 1; V[7*0 + 6, col_start+0] += 1; W[2*6 + 0, col_start+0] += 1
    U[1, col_start+1] += 1; V[7*1 + 6, col_start+1] += 1; W[2*6 + 0, col_start+1] += 1
    U[2, col_start+2] += 1; V[7*0 + 6, col_start+2] += 1; W[2*6 + 1, col_start+2] += 1
    U[3, col_start+3] += 1; V[7*1 + 6, col_start+3] += 1; W[2*6 + 1, col_start+3] += 1
    
    return U, V, W

def _construct_rank_28():
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return U, V, W

def _search_rank(T, rank, U_init, V_init, W_init, max_restarts=5, per_restart_time=15.0):
    """Search for a rank-`rank` decomposition using least_squares with multiple restarts."""
    n, m, p = 2, 2, 7
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    T_flat = T.flatten()
    
    def residuals(x):
        U = x[:dim_u*rank].reshape(dim_u, rank)
        V = x[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
        W = x[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(rank):
            T_rec += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return T_rec.flatten() - T_flat
    
    best = None
    for restart in range(max_restarts):
        if _time_remaining() < 5.0:
            break
        # Initialization
        if U_init is not None and U_init.shape[1] >= rank:
            U0 = U_init[:, :rank].copy()
            V0 = V_init[:, :rank].copy()
            W0 = W_init[:, :rank].copy()
        else:
            U0 = np.random.randn(dim_u, rank) * 0.5
            V0 = np.random.randn(dim_v, rank) * 0.5
            W0 = np.random.randn(dim_w, rank) * 0.5
        # Perturb
        scale = 0.1 if restart == 0 else 0.5
        U0 += np.random.randn(*U0.shape) * scale
        V0 += np.random.randn(*V0.shape) * scale
        W0 += np.random.randn(*W0.shape) * scale
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(residuals, x0, method='trf', max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if result.cost < 1e-8:
                x_opt = result.x
                U_opt = x_opt[:dim_u*rank].reshape(dim_u, rank)
                V_opt = x_opt[dim_u*rank:dim_u*rank+dim_v*rank].reshape(dim_v, rank)
                W_opt = x_opt[dim_u*rank+dim_v*rank:].reshape(dim_w, rank)
                # Round to half-integers
                U_r = np.round(U_opt * 2) / 2
                V_r = np.round(V_opt * 2) / 2
                W_r = np.round(W_opt * 2) / 2
                if _verify(U_r, V_r, W_r, T):
                    return U_r, V_r, W_r
                # Also check unrounded
                if _verify(U_opt, V_opt, W_opt, T) and _is_half_integer(U_opt) and _is_half_integer(V_opt) and _is_half_integer(W_opt):
                    return U_opt, V_opt, W_opt
        except Exception:
            pass
    return None

def construct():
    if _time_remaining() < 5.0:
        U, V, W = _construct_rank_28()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    T = _get_target_tensor()
    
    # Rank 25 (known valid)
    U25, V25, W25 = _construct_rank_25()
    if not _verify(U25, V25, W25, T):
        # Should not happen, but fallback
        U28, V28, W28 = _construct_rank_28()
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Try rank 24
    if _time_remaining() > 40.0:
        res24 = _search_rank(T, 24, U25, V25, W25, max_restarts=6, per_restart_time=15.0)
        if res24 is not None:
            U, V, W = res24
            if _verify(U, V, W, T):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U25.tolist(), "V": V25.tolist(), "W": W25.tolist()}
