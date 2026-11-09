import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 6]
_DEADLINE_SECONDS = 140  # Safety margin

def _residual_factory(n, m, p, R):
    """Create a residual function for the tensor decomposition."""
    def residual(x):
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:2*n*m*R].reshape(m*p, R)
        W = x[2*n*m*R:].reshape(p*n, R)
        
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        return (T_calc - 1.0).flatten()
    return residual

def _try_polish(x0, residual, max_nfev=2000):
    """Try to polish a near-solution to exact convergence."""
    try:
        res = least_squares(residual, x0, method='trf', x_scale='jac', 
                           max_nfev=max_nfev, ftol=1e-15, xtol=1e-15, gtol=1e-15)
        return res
    except Exception:
        return None

def _verify_half_integer(U, V, W, n, m, p, R):
    """Verify that U,V,W with half-integer entries give exact tensor decomposition."""
    U_int = np.round(U * 2).astype(int)
    V_int = np.round(V * 2).astype(int)
    W_int = np.round(W * 2).astype(int)
    
    U_t = U_int.reshape(n, m, R)
    V_t = V_int.reshape(m, p, R)
    W_t = W_int.reshape(p, n, R)
    
    T_calc = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    return np.all(T_calc == 8)  # 2^3 * 1 = 8

def construct():
    t0 = time.time()
    n, m, p = 3, 3, 3  # Sub-problem dimensions
    
    # Target tensor for 3x3x3: T[i, j, k] = 1
    T_target = np.ones((n, m, p))
    
    best_rank_found = 27  # Default to schoolbook equivalent
    best_U = None
    best_V = None
    best_W = None
    
    # We will try to find a solution for R in [23, 24, 25, 26]
    # If we find one, we stop.
    
    for R in range(23, 27):
        if time.time() - t0 > _DEADLINE_SECONDS - 20:  # Leave 20s for construction and fallback
            break
            
        n_vars = 3 * (n*m) * R
        residual = _residual_factory(n, m, p, R)
        
        found = False
        restart_count = 0
        
        # Adaptive restart loop: keep going until deadline or found
        while True:
            if time.time() - t0 > _DEADLINE_SECONDS - 20:
                break
            if restart_count >= 50:  # Cap restarts per rank
                break
            restart_count += 1
            
            # Structured initialization: start near schoolbook and project
            if restart_count == 1:
                # Initialize with a structured pattern
                x0 = np.zeros(n_vars)
                # Set up a basic structure
                U_init = np.zeros((n*m, R))
                V_init = np.zeros((m*p, R))
                W_init = np.zeros((p*n, R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if i*m+j < R:
                                U_init[i*m+j, i*m+j] = 1.0
                            if j*p+k < R:
                                V_init[j*p+k, j*p+k] = 1.0
                            if k*n+i < R:
                                W_init[k*n+i, k*n+i] = 1.0
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            else:
                # Random initialization with various scales
                scale = 0.5 + np.random.rand() * 1.0
                x0 = np.random.uniform(-scale, scale, n_vars)
            
            try:
                # First pass: moderate effort
                res = least_squares(residual, x0, method='trf', x_scale='jac', max_nfev=300)
                
                # If we got close, try to polish
                if res.cost < 1e-6:
                    res2 = _try_polish(res.x, residual, max_nfev=2000)
                    if res2 is not None and res2.cost < 1e-12:
                        res = res2
                
                # Check cost
                if res.cost < 1e-10:
                    # Extract U, V, W
                    U_333 = res.x[:n*m*R].reshape(n*m, R)
                    V_333 = res.x[n*m*R:2*n*m*R].reshape(m*p, R)
                    W_333 = res.x[2*n*m*R:].reshape(p*n, R)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_333 * 2) / 2
                    V_rounded = np.round(V_333 * 2) / 2
                    W_rounded = np.round(W_333 * 2) / 2
                    
                    # Verify exactness
                    if _verify_half_integer(U_rounded, V_rounded, W_rounded, n, m, p, R):
                        best_rank_found = R
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
            
    if best_U is not None and best_rank_found < 27:
        # Construct 3x3x6 decomposition
        n_full, m_full, p_full = 3, 3, 6
        R_total = 2 * best_rank_found
        
        U_total = np.zeros((n_full*m_full, R_total))
        V_total = np.zeros((m_full*p_full, R_total))
        W_total = np.zeros((p_full*n_full, R_total))
        
        # First half of B (k=0,1,2) uses first R columns
        # Second half of B (k=3,4,5) uses next R columns
        
        # U is the same for both halves (depends on i, j)
        U_total[:, 0:best_rank_found] = best_U
        U_total[:, best_rank_found:2*best_rank_found] = best_U
        
        # V: rows are j*p_full + k. p_full=6.
        for j in range(3):
            for k_prime in range(3):
                idx_v1 = j*6 + k_prime
                idx_v333 = j*3 + k_prime
                V_total[idx_v1, 0:best_rank_found] = best_V[idx_v333]
                
                idx_v2 = j*6 + (3 + k_prime)
                V_total[idx_v2, best_rank_found:2*best_rank_found] = best_V[idx_v333]
        
        # W: rows are k*n_full + i. n_full=3.
        for k_prime in range(3):
            for i in range(3):
                idx_w1 = k_prime*3 + i
                idx_w333 = k_prime*3 + i
                W_total[idx_w1, 0:best_rank_found] = best_W[idx_w333]
                
                idx_w2 = (3 + k_prime)*3 + i
                W_total[idx_w2, best_rank_found:2*best_rank_found] = best_W[idx_w333]
        
        # Final check
        U_t_full = U_total.reshape(3, 3, R_total)
        V_t_full = V_total.reshape(3, 6, R_total)
        W_t_full = W_total.reshape(6, 3, R_total)
        T_calc_full = np.einsum('ijr,jkr,kir->ijk', U_t_full, V_t_full, W_t_full)
        T_target_full = np.ones((3, 3, 6))
        
        if np.allclose(T_calc_full, T_target_full, atol=1e-9):
            return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
        else:
            # If verification failed, fall through to schoolbook
            pass
            
    # Fallback to schoolbook
    n, m, p = 3, 3, 6
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
