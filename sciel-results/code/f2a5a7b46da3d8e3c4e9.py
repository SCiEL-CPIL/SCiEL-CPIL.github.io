import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    u_rows = n * m
    v_rows = m * p
    w_rows = p * n
    
    # Build the target tensor T
    T = np.zeros((u_rows, v_rows, w_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    T_flat = T.flatten()

    def compute_residual(U, V, W):
        # U: u_rows x R
        # V: v_rows x R
        # W: w_rows x R
        # Compute sum_c U[u,c] V[v,c] W[w,c]
        # Use einsum for speed
        try:
            calc = np.einsum('uc,vc,wc->uvw', U, V, W)
        except Exception:
            # Fallback for very large tensors or errors
            calc = np.zeros_like(T)
            for c in range(U.shape[1]):
                calc += np.einsum('u,v,w->uvw', U[:,c], V[:,c], W[:,c])
        return (calc - T).flatten()

    def residual_func(params):
        R = len(params) // (u_rows + v_rows + w_rows)
        if R == 0:
            return np.zeros(len(T_flat))
        u_flat, v_flat, w_flat = params[:u_rows*R], params[u_rows*R:u_rows*R + v_rows*R], params[u_rows*R + v_rows*R:]
        U = u_flat.reshape((u_rows, R))
        V = v_flat.reshape((v_rows, R))
        W = w_flat.reshape((w_rows, R))
        return compute_residual(U, V, W)

    def check_solution(U, V, W):
        # Check if U, V, W are close to half-integers
        # and if they exactly decompose T
        # First, check half-integer property
        # We will round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check if rounding changed the values significantly? 
        # The prompt asks for integer/half-integer entries.
        # If the optimal solution is not half-integer, we might fail.
        # But we assume the minimal rank solution has such properties.
        
        # Verify exact decomposition with rounded values
        # Using integer arithmetic if possible? No, floats are fine for verification if exact.
        # But "exactly" means no floating point error.
        # Since entries are multiples of 0.5, products are multiples of 0.25.
        # Sums are multiples of 0.25.
        # T entries are 0 or 1.
        # So we can check if calc[u,v,w] is 0 or 1.
        
        try:
            calc = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r)
        except Exception:
            return False

        # Check if calc matches T exactly
        # Due to floating point, check with tolerance 1e-9
        if not np.allclose(calc, T, atol=1e-9):
            return False
            
        # Additionally, ensure the values are effectively integers or half-integers 
        # and the result is exact.
        # If allclose passes, it's likely good.
        
        # Return the rounded matrices
        return True, U_r, V_r, W_r

    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety/cleanup
    
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Initialize with schoolbook solution as fallback
    U_school = np.zeros((u_rows, best_rank))
    V_school = np.zeros((v_rows, best_rank))
    W_school = np.zeros((w_rows, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find lower ranks
    # Start from a plausible lower bound
    # M(3,3,5) is likely < 45.
    # Try 38, 35, 32, 30...
    
    ranks_to_try = [38, 36, 34, 32, 30, 28]
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart * 100 + R)
            
            # Random initialization
            # Scale: 1/sqrt(R) is often good
            scale = 1.0 / np.sqrt(R)
            U_init = np.random.randn(u_rows, R) * scale
            V_init = np.random.randn(v_rows, R) * scale
            W_init = np.random.randn(w_rows, R) * scale
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(residual_func, params_init, method='lm', max_nfev=2000)
                
                # Extract solution
                u_flat, v_flat, w_flat = res.x[:u_rows*R], res.x[u_rows*R:u_rows*R + v_rows*R], res.x[u_rows*R + v_rows*R:]
                U_sol = u_flat.reshape((u_rows, R))
                V_sol = v_flat.reshape((v_rows, R))
                W_sol = w_flat.reshape((w_rows, R))
                
                # Check if it's a valid half-integer solution
                valid, U_r, V_r, W_r = check_solution(U_sol, V_sol, W_sol)
                
                if valid:
                    best_rank = R
                    best_U, best_V, best_W = U_r, V_r, W_r
                    # Found a solution for this rank, try lower rank
                    break 
            except Exception:
                pass

    # Convert to list of lists
    # Ensure entries are floats or ints? "integer/half-integer"
    # Lists of floats are fine.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
