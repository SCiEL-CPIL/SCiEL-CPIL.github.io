"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def _build_index_maps(n, m, p):
    """Build index arrays for fast vectorized tensor evaluation."""
    I = np.repeat(np.arange(n), m * p)
    J = np.tile(np.repeat(np.arange(m), p), n)
    K = np.tile(np.arange(p), n * m)
    u_rows = I * m + J
    v_rows = J * p + K
    w_rows = K * n + I
    return u_rows, v_rows, w_rows

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # safety margin
    
    # Precompute index maps
    u_rows, v_rows, w_rows = _build_index_maps(n, m, p)
    N = n * m * p
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    def verify(U, V, W, tol=1e-8):
        """Check if decomposition is exact."""
        U_sel = U[u_rows, :]
        V_sel = V[v_rows, :]
        W_sel = W[w_rows, :]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.max(np.abs(vals - 1.0))
    
    def fast_residual(x, r):
        """Vectorized residual computation."""
        nU = n * m * r
        nV = m * p * r
        U = x[:nU].reshape(n*m, r)
        V = x[nU:nU+nV].reshape(m*p, r)
        W = x[nU+nV:].reshape(p*n, r)
        
        U_sel = U[u_rows, :]
        V_sel = V[v_rows, :]
        W_sel = W[w_rows, :]
        
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return vals - 1.0
    
    def fast_jacobian(x, r):
        """Analytical Jacobian."""
        nU = n * m * r
        nV = m * p * r
        nW = p * n * r
        n_vars = nU + nV + nW
        
        U = x[:nU].reshape(n*m, r)
        V = x[nU:nU+nV].reshape(m*p, r)
        W = x[nU+nV:].reshape(p*n, r)
        
        U_sel = U[u_rows, :]
        V_sel = V[v_rows, :]
        W_sel = W[w_rows, :]
        
        J = np.zeros((N, n_vars))
        
        dU = V_sel * W_sel
        dV = U_sel * W_sel
        dW = U_sel * V_sel
        
        ar = np.arange(r)
        for t in range(r):
            u_idx = u_rows * r + t
            v_idx = v_rows * r + t + nU
            w_idx = w_rows * r + t + nU + nV
            J[np.arange(N), u_idx] = dU[:, t]
            J[np.arange(N), v_idx] = dV[:, t]
            J[np.arange(N), w_idx] = dW[:, t]
            
        return J
    
    def compress_factors(U, V, W, r, max_rounds=20):
        """
        Greedy basis-reduction with local re-optimization.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        cur_r = r
        
        for _ in range(max_rounds):
            if time.time() > deadline - 15:
                break
            
            best_remove = -1
            best_cost = np.inf
            best_new_U, best_new_V, best_new_W = None, None, None
            
            for t in range(cur_r):
                if time.time() > deadline - 15:
                    break
                
                mask = np.ones(cur_r, dtype=bool)
                mask[t] = False
                new_r = cur_r - 1
                
                U_red = U[:, mask]
                V_red = V[:, mask]
                W_red = W[:, mask]
                
                # Try to re-optimize the reduced system
                x_flat = np.concatenate([U_red.flatten(), V_red.flatten(), W_red.flatten()])
                
                try:
                    result = least_squares(
                        fast_residual, x_flat,
                        jac=lambda x: fast_jacobian(x, new_r),
                        args=(new_r,),
                        method='trf',
                        max_nfev=200,
                        ftol=1e-10,
                        xtol=1e-10,
                        gtol=1e-10
                    )
                    
                    if result.cost < 1e-8:
                        sol = result.x
                        nU_new = n * m * new_r
                        nV_new = m * p * new_r
                        U_new = sol[:nU_new].reshape(n*m, new_r)
                        V_new = sol[nU_new:nU_new+nV_new].reshape(m*p, new_r)
                        W_new = sol[nU_new+nV_new:].reshape(p*n, new_r)
                        
                        err = verify(U_new, V_new, W_new)
                        if err < 1e-7:
                            if err < best_cost:
                                best_cost = err
                                best_remove = t
                                best_new_U, best_new_V, best_new_W = U_new, V_new, W_new
                except Exception:
                    pass
            
            if best_remove >= 0 and best_new_U is not None:
                U, V, W = best_new_U, best_new_V, best_new_W
                cur_r -= 1
            else:
                break
        
        return U, V, W, cur_r
    
    def polish_solution(U, V, W, r):
        """High-precision polish pass."""
        x_flat = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        try:
            result = least_squares(
                fast_residual, x_flat,
                jac=lambda x: fast_jacobian(x, r),
                args=(r,),
                method='trf',
                max_nfev=500,
                ftol=1e-14,
                xtol=1e-14,
                gtol=1e-14
            )
            if result.cost < 1e-12:
                sol = result.x
                nU = n * m * r
                nV = m * p * r
                U_new = sol[:nU].reshape(n*m, r)
                V_new = sol[nU:nU+nV].reshape(m*p, r)
                W_new = sol[nU+nV:].reshape(p*n, r)
                if verify(U_new, V_new, W_new) < 1e-10:
                    return U_new, V_new, W_new
        except Exception:
            pass
        return U, V, W, r
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But realistically, for 3x4x7, known bounds suggest rank around 40-60
    min_rank_to_try = 30
    
    np.random.seed(42)
    
    for r in range(min_rank_to_try, rank_school + 1):
        if time.time() > deadline - 25:
            break
        
        n_vars = r * (n*m + m*p + p*n)
        
        found = False
        n_inits = 8
        
        for init in range(n_inits):
            if time.time() > deadline - 20:
                break
            
            # Different initialization strategies
            if init == 0:
                # Pure random
                x0 = np.random.randn(n_vars) * 0.5
            elif init == 1:
                # Random with larger scale
                x0 = np.random.randn(n_vars) * 1.0
            elif init == 2:
                # Structured: partial schoolbook pattern
                x0 = np.zeros(n_vars)
                # Use first min(r, rank_school) schoolbook columns as basis
                n_use = min(r, rank_school)
                for c in range(n_use):
                    i = c // (m*p)
                    j = (c // p) % m
                    k = c % p
                    # U component
                    u_idx = (i*m+j) * r + (c % r)
                    x0[u_idx] = 1.0
                    # V component
                    v_idx = n*m*r + (j*p+k) * r + (c % r)
                    x0[v_idx] = 1.0
                    # W component
                    w_idx = n*m*r + m*p*r + (k*n+i) * r + (c % r)
                    x0[w_idx] = 1.0
                # Add small noise
                x0 += 0.1 * np.random.randn(n_vars)
            elif init == 3:
                # Random but with zero-mean columns
                U0 = np.random.randn(n*m, r)
                V0 = np.random.randn(m*p, r)
                W0 = np.random.randn(p*n, r)
                # Normalize columns
                for c in range(r):
                    nU_c = np.linalg.norm(U0[:, c])
                    nV_c = np.linalg.norm(V0[:, c])
                    nW_c = np.linalg.norm(W0[:, c])
                    if nU_c > 1e-10:
                        U0[:, c] /= nU_c
                    if nV_c > 1e-10:
                        V0[:, c] /= nV_c
                    if nW_c > 1e-10:
                        W0[:, c] /= nW_c
                x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            else:
                # Random with varying scales
                scale = 0.3 + 0.2 * init
                x0 = np.random.randn(n_vars) * scale
            
            try:
                result = least_squares(
                    fast_residual, x0,
                    jac=lambda x: fast_jacobian(x, r),
                    args=(r,),
                    method='trf',
                    max_nfev=800,
                    ftol=1e-11,
                    xtol=1e-11,
                    gtol=1e-11
                )
                
                if result.cost < 1e-9:
                    sol = result.x
                    nU = n * m * r
                    nV = m * p * r
                    U_sol = sol[:nU].reshape(n*m, r)
                    V_sol = sol[nU:nU+nV].reshape(m*p, r)
                    W_sol = sol[nU+nV:].reshape(p*n, r)
                    
                    max_err = verify(U_sol, V_sol, W_sol)
                    
                    if max_err < 1e-7:
                        # Polish
                        U_pol, V_pol, W_pol, _ = polish_solution(U_sol, V_sol, W_sol, r)
                        
                        # Try compression
                        U_comp, V_comp, W_comp, comp_r = compress_factors(U_pol, V_pol, W_pol, r)
                        
                        if comp_r < r:
                            max_err_comp = verify(U_comp, V_comp, W_comp)
                            if max_err_comp < 1e-7:
                                best_rank = comp_r
                                best_U, best_V, best_W = U_comp, V_comp, W_comp
                            else:
                                best_rank = r
                                best_U, best_V, best_W = U_pol, V_pol, W_pol
                        else:
                            best_rank = r
                            best_U, best_V, best_W = U_pol, V_pol, W_pol
                        
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    # Final verification
    max_err = verify(best_U, best_V, best_W)
    if max_err > 1e-6:
        # Fall back to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_school
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
