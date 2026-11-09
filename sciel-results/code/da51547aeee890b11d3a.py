import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    n_rows, n_cols, n_depth = T.shape
    
    # Try ranks from low to high
    min_rank = max(n * m, m * p, p * n)  # 15
    schoolbook_rank = n * m * p  # 30
    
    for rank in range(min_rank, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(25):
            if time.time() > deadline - 5:
                break
            
            # ALS initialization
            result = _als_decompose(T, rank, max_iter=100, tol=1e-10, seed=attempt)
            if result is not None:
                U, V, W = result
            else:
                rng = np.random.RandomState(attempt)
                U = rng.randn(n_rows, rank)
                V = rng.randn(n_cols, rank)
                W = rng.randn(n_depth, rank)
            
            # Polish with Levenberg-Marquardt
            try:
                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                result_lm = least_squares(_tensor_residual, x0, args=(T, n_rows, n_cols, n_depth, rank), 
                                         method='lm', max_nfev=500)
                x = result_lm.x
                U_lm = x[:n_rows*rank].reshape(n_rows, rank)
                V_lm = x[n_rows*rank:n_rows*rank+n_cols*rank].reshape(n_cols, rank)
                W_lm = x[n_rows*rank+n_cols*rank:].reshape(n_depth, rank)
                
                T_recon = np.einsum('ic,jc,kc->ijk', U_lm, V_lm, W_lm)
                err = np.max(np.abs(T_recon - T))
                if err < 1e-9:
                    # Snap to half-integers
                    U_snap = np.round(U_lm * 2) / 2
                    V_snap = np.round(V_lm * 2) / 2
                    W_snap = np.round(W_lm * 2) / 2
                    T_recon2 = np.einsum('ic,jc,kc->ijk', U_snap, V_snap, W_snap)
                    err2 = np.max(np.abs(T_recon2 - T))
                    if err2 < 1e-9:
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    else:
                        # Try integers
                        U_int = np.round(U_lm)
                        V_int = np.round(V_lm)
                        W_int = np.round(W_lm)
                        T_recon3 = np.einsum('ic,jc,kc->ijk', U_int, V_int, W_int)
                        err3 = np.max(np.abs(T_recon3 - T))
                        if err3 < 1e-9:
                            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        else:
                            return {"U": U_lm.tolist(), "V": V_lm.tolist(), "W": W_lm.tolist()}
            except Exception:
                pass
    
    # Fallback: schoolbook
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


def _tensor_residual(x, T, n_rows, n_cols, n_depth, rank):
    U = x[:n_rows*rank].reshape(n_rows, rank)
    V = x[n_rows*rank:n_rows*rank+n_cols*rank].reshape(n_cols, rank)
    W = x[n_rows*rank+n_cols*rank:].reshape(n_depth, rank)
    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return (T_recon - T).flatten()


def _als_decompose(T, rank, max_iter=100, tol=1e-10, seed=0):
    """ALS for CP decomposition of a 3-way tensor."""
    rng = np.random.RandomState(seed)
    n_rows, n_cols, n_depth = T.shape
    
    # Initialize with random values
    U = rng.randn(n_rows, rank)
    V = rng.randn(n_cols, rank)
    W = rng.randn(n_depth, rank)
    
    prev_err = np.inf
    for it in range(max_iter):
        # Update U
        for c in range(rank):
            v_c = V[:, c]
            w_c = W[:, c]
            vec = np.einsum('ijk,j,k->i', T, v_c, w_c)
            denom = np.dot(v_c, v_c) * np.dot(w_c, w_c)
            if abs(denom) > 1e-15:
                U[:, c] = vec / denom
        
        # Update V
        for c in range(rank):
            u_c = U[:, c]
            w_c = W[:, c]
            vec = np.einsum('ijk,i,k->j', T, u_c, w_c)
            denom = np.dot(u_c, u_c) * np.dot(w_c, w_c)
            if abs(denom) > 1e-15:
                V[:, c] = vec / denom
        
        # Update W
        for c in range(rank):
            u_c = U[:, c]
            v_c = V[:, c]
            vec = np.einsum('ijk,i,j->k', T, u_c, v_c)
            denom = np.dot(u_c, u_c) * np.dot(v_c, v_c)
            if abs(denom) > 1e-15:
                W[:, c] = vec / denom
        
        # Check convergence
        if it > 0:
            T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < tol:
                return U, V, W
            if abs(prev_err - err) < tol:
                if err < 1e-9:
                    return U, V, W
                break
            prev_err = err
    
    # Final check
    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    err = np.max(np.abs(T_recon - T))
    if err < 1e-9:
        return U, V, W
    return None

# EVOLVE-BLOCK-END
