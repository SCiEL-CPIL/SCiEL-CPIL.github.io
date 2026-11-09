import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 6]
_DEADLINE_SECONDS = 140  # Safety margin

def construct():
    t0 = time.time()
    n, m, p = 3, 3, 3  # Sub-problem dimensions
    
    # Target tensor for 3x3x3: T[i, j, k] = 1
    T_target = np.ones((n, m, p))
    
    # We want to find U, V, W of rank R such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # U shape: (n*m, R)
    # V shape: (m*p, R)
    # W shape: (p*n, R)
    
    # Try ranks from 23 up to 26?
    # If we find a solution for R, the total rank for 3x3x6 will be 2*R.
    # Schoolbook is 54 (which is 2*27).
    # So if we find R < 27, we improve.
    
    best_rank_found = 27 # Default to schoolbook equivalent
    best_U = None
    best_V = None
    best_W = None
    
    # We will try to find a solution for R in [23, 24, 25, 26]
    # If we find one, we stop.
    
    for R in range(23, 27):
        if time.time() - t0 > _DEADLINE_SECONDS - 20: # Leave 20s for construction and fallback
            break
            
        n_vars = 3 * (n*m) * R
        # Note: U, V, W all have n*m = 9 rows.
        
        found = False
        # Try a few restarts
        for restart in range(10):
            if time.time() - t0 > _DEADLINE_SECONDS - 20:
                break
                
            x0 = np.random.randn(n_vars)
            # Scale x0 to be reasonable, e.g., uniform [-1, 1]
            x0 = np.random.uniform(-1, 1, n_vars)
            
            def residual(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R:2*n*m*R].reshape(m*p, R)
                W = x[2*n*m*R:].reshape(p*n, R)
                
                U_t = U.reshape(n, m, R)
                V_t = V.reshape(m, p, R)
                W_t = W.reshape(p, n, R)
                
                T_calc = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                
                return (T_calc - T_target).flatten()
            
            try:
                res = least_squares(residual, x0, method='trf', x_scale='jac', max_nfev=500)
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
                    
                    # Verify exactness with integer arithmetic
                    # Scale by 2 to get integers
                    U_int = np.round(U_rounded * 2).astype(int)
                    V_int = np.round(V_rounded * 2).astype(int)
                    W_int = np.round(W_rounded * 2).astype(int)
                    
                    U_t_int = U_int.reshape(n, m, R)
                    V_t_int = V_int.reshape(m, p, R)
                    W_t_int = W_int.reshape(p, n, R)
                    
                    T_calc_int = np.einsum('ijr,jkr,kir->ijk', U_t_int, V_t_int, W_t_int)
                    
                    # We want T_calc_int == 8 (since 2^3 * 1 = 8)
                    if np.all(T_calc_int == 8):
                        # Found a valid solution!
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
        # k=0,1,2 -> indices 0..5, 6..11, 12..17
        # k=3,4,5 -> indices 18..23, 24..29, 30..35
        # V_333 has rows j*3 + k' where k'=0,1,2.
        
        for j in range(3):
            for k_prime in range(3):
                # First block
                idx_v1 = j*6 + k_prime
                idx_v333 = j*3 + k_prime
                V_total[idx_v1, 0:best_rank_found] = best_V[idx_v333]
                
                # Second block
                idx_v2 = j*6 + (3 + k_prime)
                V_total[idx_v2, best_rank_found:2*best_rank_found] = best_V[idx_v333]
        
        # W: rows are k*n_full + i. n_full=3.
        # k=0,1,2 -> indices 0..2, 3..5, 6..8
        # k=3,4,5 -> indices 9..11, 12..14, 15..17
        # W_333 has rows k'*3 + i.
        
        for k_prime in range(3):
            for i in range(3):
                # First block
                idx_w1 = k_prime*3 + i
                idx_w333 = k_prime*3 + i
                W_total[idx_w1, 0:best_rank_found] = best_W[idx_w333]
                
                # Second block
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
