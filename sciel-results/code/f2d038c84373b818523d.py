import time
import numpy as np
from scipy.optimize import minimize, linprog

DIMENSIONS = [2, 2, 9]

def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[i, j, k] = 1.0
    return T

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        # Strassen 7 products for 2x2 * 2x2 (columns k0, k1)
        # M1 = (A11+A22)(B11+B22)
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M2 = A22(B21-B11)
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        # M3 = A11(B12-B22)
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M4 = A22(B11+B22)
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        # M5 = (A11+A12)B22
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        # M6 = (A21-A11)(B11+B12)
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        # M7 = (A12-A22)(B21+B22)
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute max absolute residual."""
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    T_true = _build_true_tensor(n, m, p)
    return np.max(np.abs(M - T_true))

def _try_linear_reduction(U, V, W, n, m, p, deadline):
    """Try to reduce rank by 1 via linear algebra."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    T_true = _build_true_tensor(n, m, p)
    
    # Flatten tensors for linear system
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # Fix V,W: for each i, T[i,:,:].flatten() = U[i,:] @ (V*W).T reshaped
    # Actually: T[i,j,k] = sum_r U[i,r] * (V[j,r]*W[k,r])
    # So for fixed i: T[i,:,:].flatten() = U[i,:] @ M_i where M_i[r, j*p+k] = V[j,r]*W[k,r]
    
    # Precompute V*W outer products
    # VW[r, j, k] = V[j,r] * W[k,r]
    VW = np.einsum('jr,kr->rjk', V, W)  # (rank, mp, pn)
    VW_flat = VW.transpose(1, 2, 0).reshape(mp * pn, rank)  # (mp*pn, rank)
    
    # Fix V, W, solve for U
    # T[i,:,:].flatten() = U[i,:] @ VW_flat.T  =>  T_flat[i] = U[i,:] @ VW_flat.T
    # So: U[i,:] = T_flat[i] @ (VW_flat.T)^+ 
    # Or: solve VW_flat @ U.T = T_true_flat.T
    
    T_flat = T_true.reshape(nm, -1)  # (nm, mp*pn)
    
    for c in range(rank):
        if time.time() > deadline:
            break
        idx = [i for i in range(rank) if i != c]
        VW_sub = VW_flat[:, idx]  # (mp*pn, rank-1)
        
        # Solve: U_sub @ VW_sub.T = T_flat  =>  VW_sub @ U_sub.T = T_flat.T
        # U_sub is (nm, rank-1)
        try:
            # Use lstsq: VW_sub @ X = T_flat.T where X = U_sub.T (rank-1, nm)
            X, residuals, rank_s, sv = np.linalg.lstsq(VW_sub, T_flat.T, rcond=None)
            U_sub = X.T  # (nm, rank-1)
            
            # Check residual
            U_new = np.zeros((nm, rank))
            U_new[:, idx] = U_sub
            res = _compute_residual(U_new, V, W, n, m, p)
            if res < 1e-8:
                return U_new[:, idx], V[:, idx], W[:, idx]
        except:
            continue
    
    # Fix U, W, solve for V
    UW = np.einsum('ir,kr->rik', U, W)  # (rank, nm, pn)
    UW_flat = UW.transpose(1, 2, 0).reshape(nm * pn, rank)  # (nm*pn, rank)
    
    T_flat_V = T_true.transpose(1, 0, 2).reshape(mp, -1)  # (mp, nm*pn)
    
    for c in range(rank):
        if time.time() > deadline:
            break
        idx = [i for i in range(rank) if i != c]
        UW_sub = UW_flat[:, idx]
        
        try:
            X, residuals, rank_s, sv = np.linalg.lstsq(UW_sub, T_flat_V.T, rcond=None)
            V_sub = X.T  # (mp, rank-1)
            
            V_new = np.zeros((mp, rank))
            V_new[:, idx] = V_sub
            res = _compute_residual(U, V_new, W, n, m, p)
            if res < 1e-8:
                return U[:, idx], V_new[:, idx], W[:, idx]
        except:
            continue
    
    # Fix U, V, solve for W
    UV = np.einsum('ir,jr->rij', U, V)  # (rank, nm, mp)
    UV_flat = UV.transpose(1, 2, 0).reshape(nm * mp, rank)  # (nm*mp, rank)
    
    T_flat_W = T_true.transpose(2, 0, 1).reshape(pn, -1)  # (pn, nm*mp)
    
    for c in range(rank):
        if time.time() > deadline:
            break
        idx = [i for i in range(rank) if i != c]
        UV_sub = UV_flat[:, idx]
        
        try:
            X, residuals, rank_s, sv = np.linalg.lstsq(UV_sub, T_flat_W.T, rcond=None)
            W_sub = X.T  # (pn, rank-1)
            
            W_new = np.zeros((pn, rank))
            W_new[:, idx] = W_sub
            res = _compute_residual(U, V, W_new, n, m, p)
            if res < 1e-8:
                return U[:, idx], V[:, idx], W_new[:, idx]
        except:
            continue
    
    return None

def _try_optimization_reduction(U, V, W, n, m, p, deadline):
    """Try to find rank-31 via smooth optimization."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    T_true = _build_true_tensor(n, m, p)
    
    best_result = None
    best_res = np.inf
    
    # Try removing each column and optimizing
    for c in range(rank):
        if time.time() > deadline:
            break
        
        idx = [i for i in range(rank) if i != c]
        U0 = U[:, idx].copy()
        V0 = V[:, idx].copy()
        W0 = W[:, idx].copy()
        
        r2 = rank - 1
        
        def objective(x):
            if time.time() > deadline:
                return 1e10
            U_x = x[:nm * r2].reshape(nm, r2)
            V_x = x[nm * r2: nm * r2 + mp * r2].reshape(mp, r2)
            W_x = x[nm * r2 + mp * r2:].reshape(pn, r2)
            M = np.einsum('ir,jr,kr->ijk', U_x, V_x, W_x)
            res = np.max(np.abs(M - T_true))
            return res**2
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.fun < 1e-14:
                U_x = result.x[:nm * r2].reshape(nm, r2)
                V_x = result.x[nm * r2: nm * r2 + mp * r2].reshape(mp, r2)
                W_x = result.x[nm * r2 + mp * r2:].reshape(pn, r2)
                res = _compute_residual(U_x, V_x, W_x, n, m, p)
                if res < 1e-8:
                    return U_x, V_x, W_x
                if res < best_res:
                    best_res = res
                    best_result = (U_x, V_x, W_x)
        except:
            continue
    
    return best_result if best_res < 1e-6 else None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build rank-32 solution
    U, V, W = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
    
    # Try linear reduction
    if time.time() < deadline - 10:
        result = _try_linear_reduction(U, V, W, n, m, p, deadline - 5)
        if result is not None:
            U31, V31, W31 = result
            return {"U": to_list(U31), "V": to_list(V31), "W": to_list(W31)}
    
    # Try optimization-based reduction
    if time.time() < deadline - 5:
        result = _try_optimization_reduction(U, V, W, n, m, p, deadline - 2)
        if result is not None:
            U31, V31, W31 = result
            return {"U": to_list(U31), "V": to_list(V31), "W": to_list(W31)}
    
    # Fall back to rank-32
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
