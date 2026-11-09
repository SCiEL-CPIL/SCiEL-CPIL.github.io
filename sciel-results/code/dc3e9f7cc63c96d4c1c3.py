import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 6]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s out of 120s
    
    n, m, p = 3, 3, 3 # Sub-problem
    
    # Target tensor for 3x3x3
    # T[i, j, k] = 1
    T_target = np.ones((n, m, p))
    
    # Variables for U, V, W flattened
    # Shapes: (9, 23)
    R = 23
    n_vars = 3 * (n*m) * R
    x0 = np.random.randn(n_vars)
    
    def residual(x):
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:2*n*m*R].reshape(m*p, R)
        W = x[2*n*m*R:].reshape(p*n, R)
        
        # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # einsum: 'jr,kr,ir->ijk'? No.
        # U index: i*m+j
        # V index: j*p+k
        # W index: k*n+i
        # We want T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # Reshape to 3D tensors to use einsum easily?
        # U_t: [i, j, r]
        # V_t: [j, k, r]
        # W_t: [k, i, r]
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        # Sum over r
        # T[i, j, k] = sum_r U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
        # einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        return (T_calc - T_target).flatten()
    
    best_result = None
    best_cost = np.inf
    
    for i in range(20): # 20 restarts
        if time.time() > deadline:
            break
            
        x0 = np.random.randn(n_vars)
        try:
            res = least_squares(residual, x0, method='trf', x_scale='jac', max_nfev=1000)
            if res.cost < best_cost:
                best_cost = res.cost
                best_result = res.x
        except Exception:
            pass
            
    if best_result is not None and best_cost < 1e-6:
        # Extract U, V, W
        U_333 = best_result[:n*m*R].reshape(n*m, R)
        V_333 = best_result[n*m*R:2*n*m*R].reshape(m*p, R)
        W_333 = best_result[2*n*m*R:].reshape(p*n, R)
        
        # Round to nearest 0.5
        U_333_rounded = np.round(U_333 * 2) / 2
        V_333_rounded = np.round(V_333 * 2) / 2
        W_333_rounded = np.round(W_333 * 2) / 2
        
        # Verify rounded solution
        U_t = U_333_rounded.reshape(n, m, R)
        V_t = V_333_rounded.reshape(m, p, R)
        W_t = W_333_rounded.reshape(p, n, R)
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        if np.allclose(T_calc, T_target, atol=1e-9):
            # Construct 3x3x6 decomposition
            # n=3, m=3, p=6
            # Rank = 2 * 23 = 46
            R_total = 2 * R
            U_total = np.zeros((3*3, R_total))
            V_total = np.zeros((3*6, R_total))
            W_total = np.zeros((6*3, R_total))
            
            # First 3 columns of B (k=0,1,2) use first 23 factors
            # Second 3 columns of B (k=3,4,5) use next 23 factors
            
            # U is same for both (depends on i, j)
            U_total[:, 0:R] = U_333_rounded
            U_total[:, R:2*R] = U_333_rounded
            
            # V: rows are j*p + k. p=6.
            # k=0,1,2 -> indices 0..5, 6..11, 12..17
            # k=3,4,5 -> indices 18..23, 24..29, 30..35
            # V_333 has rows j*3 + k' where k'=0,1,2.
            
            for j in range(3):
                for k_prime in range(3):
                    # First block
                    idx_v1 = j*6 + k_prime
                    idx_v333 = j*3 + k_prime
                    V_total[idx_v1, 0:R] = V_333_rounded[idx_v333]
                    
                    # Second block
                    idx_v2 = j*6 + (3 + k_prime)
                    V_total[idx_v2, R:2*R] = V_333_rounded[idx_v333]
            
            # W: rows are k*n + i. n=3.
            # k=0,1,2 -> indices 0..2, 3..5, 6..8
            # k=3,4,5 -> indices 9..11, 12..14, 15..17
            # W_333 has rows k'*3 + i.
            
            for k_prime in range(3):
                for i in range(3):
                    # First block
                    idx_w1 = k_prime*3 + i
                    idx_w333 = k_prime*3 + i
                    W_total[idx_w1, 0:R] = W_333_rounded[idx_w333]
                    
                    # Second block
                    idx_w2 = (3 + k_prime)*3 + i
                    W_total[idx_w2, R:2*R] = W_333_rounded[idx_w333]
            
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
