import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def _rank11_solution():
    """Return the known rank-11 decomposition."""
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    # 1. (A00+A11)(B00+B11) -> M1
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # 2. (A10+A11)B00 -> M2
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # 3. A00(B01-B11) -> M3
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # 4. A11(B10-B00) -> M4
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # 5. (A00+A01)B11 -> M5
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # 6. (A10-A00)(B00+B01) -> M6
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # 7. (A01-A11)(B10+B11) -> M7
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # 8. A00 B02 -> M8
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # 9. A01 B12 -> M9
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # 10. A10 B02 -> M10
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # 11. A11 B12 -> M11
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _als_iteration(U, V, W, T, n, m, p, target_rank):
    """One iteration of alternating least squares."""
    # Update U
    X_u = np.zeros((m*p*p*n, target_rank))
    for rr in range(target_rank):
        X_u[:,rr] = np.outer(V[:,rr], W[:,rr]).flatten()
    U_new, _, _, _ = np.linalg.lstsq(X_u, T.reshape(n*m, -1).T, rcond=None)
    U_new = U_new.T
    
    # Update V
    X_v = np.zeros((n*m*p*n, target_rank))
    for rr in range(target_rank):
        X_v[:,rr] = np.outer(U_new[:,rr], W[:,rr]).flatten()
    V_new, _, _, _ = np.linalg.lstsq(X_v, T.reshape(m*p, -1).T, rcond=None)
    V_new = V_new.T
    
    # Update W
    X_w = np.zeros((n*m*m*p, target_rank))
    for rr in range(target_rank):
        X_w[:,rr] = np.outer(U_new[:,rr], V_new[:,rr]).flatten()
    W_new, _, _, _ = np.linalg.lstsq(X_w, T.reshape(p*n, -1).T, rcond=None)
    W_new = W_new.T
    
    return U_new, V_new, W_new

def _compute_error(U, V, W, T, target_rank):
    """Compute reconstruction error."""
    T_rec = np.zeros_like(T)
    for rr in range(target_rank):
        T_rec += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
    return np.linalg.norm(T - T_rec)

def _check_half_integer(U, V, W, tol=1e-3):
    """Check if all entries are half-integers."""
    for mat in [U, V, W]:
        for val in mat:
            nearest = round(val * 2) / 2
            if abs(val - nearest) > tol:
                return False
    return True

def _try_rank10_search(n, m, p, T, deadline):
    """Try to find a rank-10 decomposition using structured search."""
    target_rank = 10
    U11, V11, W11 = _rank11_solution()
    
    # Strategy 1: Drop each term and perturb with different scales
    for drop_idx in range(11):
        if time.time() > deadline:
            break
        
        for pert_scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
            if time.time() > deadline:
                break
            
            U0 = np.delete(U11, drop_idx, axis=1)
            V0 = np.delete(V11, drop_idx, axis=1)
            W0 = np.delete(W11, drop_idx, axis=1)
            
            # Multiple random restarts per scale
            for restart in range(3):
                if time.time() > deadline:
                    break
                
                U = U0 + pert_scale * np.random.randn(n*m, target_rank)
                V = V0 + pert_scale * np.random.randn(m*p, target_rank)
                W = W0 + pert_scale * np.random.randn(p*n, target_rank)
                
                # Run ALS with adaptive iterations
                best_err = float('inf')
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                for it in range(300):
                    if time.time() > deadline:
                        break
                    
                    U, V, W = _als_iteration(U, V, W, T, n, m, p, target_rank)
                    err = _compute_error(U, V, W, T, target_rank)
                    
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                    # Early stop if convergence
                    if err < 1e-8:
                        break
                
                # High-precision polish pass
                if best_err < 1e-4:
                    U, V, W = best_U, best_V, best_W
                    for polish_it in range(100):
                        if time.time() > deadline:
                            break
                        U, V, W = _als_iteration(U, V, W, T, n, m, p, target_rank)
                        err = _compute_error(U, V, W, T, target_rank)
                        if err < 1e-10:
                            break
                    
                    if err < 1e-6:
                        if _check_half_integer(U, V, W):
                            U_s = np.round(U * 2) / 2
                            V_s = np.round(V * 2) / 2
                            W_s = np.round(W * 2) / 2
                            
                            # Final verification
                            T_rec = np.zeros_like(T)
                            for rr in range(target_rank):
                                T_rec += np.einsum('a,b,c->abc', U_s[:,rr], V_s[:,rr], W_s[:,rr])
                            
                            if np.allclose(T_rec, T, atol=1e-9):
                                return U_s, V_s, W_s
    
    # Strategy 2: Structured initialization preserving Strassen pattern
    # Keep first 7 terms (Strassen for k=0,1) and search for 3 terms for k=2
    if time.time() < deadline:
        U_strassen = U11[:, :7]
        V_strassen = V11[:, :7]
        W_strassen = W11[:, :7]
        
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization for the 3 additional terms
            U_extra = np.random.randn(n*m, 3) * 0.5
            V_extra = np.random.randn(m*p, 3) * 0.5
            W_extra = np.random.randn(p*n, 3) * 0.5
            
            U = np.hstack([U_strassen, U_extra])
            V = np.hstack([V_strassen, V_extra])
            W = np.hstack([W_strassen, W_extra])
            
            # Fix the Strassen part and only optimize the extra part
            # This is a constrained optimization
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Only update the last 3 columns
                U_work = U[:, 7:]
                V_work = V[:, 7:]
                W_work = W[:, 7:]
                
                # Compute residual after Strassen part
                T_res = T.copy()
                for rr in range(7):
                    T_res -= np.einsum('a,b,c->abc', U_strassen[:,rr], V_strassen[:,rr], W_strassen[:,rr])
                
                # ALS on the residual with 3 terms
                for sub_it in range(50):
                    if time.time() > deadline:
                        break
                    
                    # Update U_extra
                    X_u = np.zeros((m*p*p*n, 3))
                    for rr in range(3):
                        X_u[:,rr] = np.outer(V_work[:,rr], W_work[:,rr]).flatten()
                    U_extra, _, _, _ = np.linalg.lstsq(X_u, T_res.reshape(n*m, -1).T, rcond=None)
                    U_extra = U_extra.T
                    
                    # Update V_extra
                    X_v = np.zeros((n*m*p*n, 3))
                    for rr in range(3):
                        X_v[:,rr] = np.outer(U_extra[:,rr], W_work[:,rr]).flatten()
                    V_extra, _, _, _ = np.linalg.lstsq(X_v, T_res.reshape(m*p, -1).T, rcond=None)
                    V_extra = V_extra.T
                    
                    # Update W_extra
                    X_w = np.zeros((n*m*m*p, 3))
                    for rr in range(3):
                        X_w[:,rr] = np.outer(U_extra[:,rr], V_extra[:,rr]).flatten()
                    W_extra, _, _, _ = np.linalg.lstsq(X_w, T_res.reshape(p*n, -1).T, rcond=None)
                    W_extra = W_extra.T
                
                U[:, 7:] = U_extra
                V[:, 7:] = V_extra
                W[:, 7:] = W_extra
                
                err = _compute_error(U, V, W, T, target_rank)
                if err < 1e-8:
                    break
            
            if err < 1e-6:
                if _check_half_integer(U, V, W):
                    U_s = np.round(U * 2) / 2
                    V_s = np.round(V * 2) / 2
                    W_s = np.round(W * 2) / 2
                    
                    T_rec = np.zeros_like(T)
                    for rr in range(target_rank):
                        T_rec += np.einsum('a,b,c->abc', U_s[:,rr], V_s[:,rr], W_s[:,rr])
                    
                    if np.allclose(T_rec, T, atol=1e-9):
                        return U_s, V_s, W_s
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Try structured rank-10 search
    result = _try_rank10_search(n, m, p, T, deadline)
    
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-11
    U, V, W = _rank11_solution()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
