import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Target Rank
    target_rank = 14
    
    # Initialize U, V, W with random small integers or floats
    # We want to find a solution where U, V, W are integers or half-integers.
    # We can solve for floats first, then check for rounding.
    
    best_result = None
    best_score = n * m * p  # Schoolbook rank
    
    # Generate Schoolbook as fallback
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    # Try to find rank 14 decomposition
    # Variables: U (9x14), V (15x14), W (15x14)
    # Flattened: 9*14 + 15*14 + 15*14 = 126 + 210 + 210 = 546 variables
    
    # We use a randomized restart approach with a local solver
    # Since the system is underdetermined, we can fix some variables or use random init.
    
    def residuals(params, U_shape, V_shape, W_shape):
        u_len = U_shape[0] * U_shape[1]
        v_len = V_shape[0] * V_shape[1]
        w_len = W_shape[0] * W_shape[1]
        
        U = params[:u_len].reshape(U_shape)
        V = params[u_len:u_len+v_len].reshape(V_shape)
        W = params[u_len+v_len:].reshape(W_shape)
        
        # Tensor product sum
        # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
        # We want T[i, j, k] == 1 for all i, j, k
        
        # Efficient calculation:
        # For each c, outer product of U[:,c] and V[:,c] gives a (n*m) x (m*p) matrix? No.
        # U is (n*m) x R. V is (m*p) x R. W is (p*n) x R.
        # Term c contributes to T[i, j, k] via U[i, c] * V[j, c] * W[k, c].
        # Let's compute the tensor T as a (n, m, p) array.
        
        T = np.zeros((n, m, p))
        for c in range(U_shape[1]):
            u_col = U[:, c].reshape(n, m)
            v_col = V[:, c].reshape(m, p)
            w_col = W[:, c].reshape(p, n)
            
            # u_col is n x m. v_col is m x p.
            # The term is u_col[i, j] * v_col[j, k] * w_col[k, i]?
            # Wait, the tensor index for W is k*n + i. So W[k*n+i, c] is the coefficient.
            # The standard matrix mult tensor is T[i, j, k] = 1.
            # Decomposition: T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
            
            # So for a fixed c:
            # Contribution to T[i, j, k] is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
            
            # We can compute this by iterating i, j, k or using einsum.
            # Let's use einsum for speed if possible, but loop over c is fine for R=14.
            
            # u_col is flattened U[:, c].
            # v_col is flattened V[:, c].
            # w_col is flattened W[:, c].
            
            # Reshape w_col to (p, n)
            # w_col reshaped to (p, n) means w_col[k*n + i] is w_reshaped[k, i].
            
            # We want sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1.
            
            # Let's just compute the full tensor for this c and add it.
            # T_c[i, j, k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # This is a rank-1 tensor in the (i, j, k) space?
            # No, the indices are coupled.
            # U depends on i, j. V depends on j, k. W depends on k, i.
            # This is not a simple outer product of three vectors.
            
            # So we must compute it carefully.
            # For a fixed c, let u_c be the 9-vector, v_c be the 15-vector, w_c be the 15-vector.
            # We can loop over i, j, k. 45 iterations.
            
            for i in range(n):
                for j in range(m):
                    u_val = U[i*m + j, c]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = V[j*p + k, c]
                        if v_val == 0: continue
                        w_val = W[k*n + i, c]
                        T[i, j, k] += u_val * v_val * w_val
                        
        # Residuals are T[i, j, k] - 1
        return (T - 1).flatten()
    
    U_shape = (n*m, target_rank)
    V_shape = (m*p, target_rank)
    W_shape = (p*n, target_rank)
    
    # Try a few random initializations
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Random initialization with small integers
        # Use integers in {-1, 0, 1} or {-2, -1, 0, 1, 2}
        # Actually, random floats might converge better to the integer solution.
        params = np.random.uniform(-2, 2, size=(U_shape[0]*U_shape[1] + V_shape[0]*V_shape[1] + W_shape[0]*W_shape[1]))
        
        try:
            # Use scipy least_squares
            from scipy.optimize import least_squares
            result = least_squares(residuals, params, args=(U_shape, V_shape, W_shape), method='trf', max_nfev=5000, xtol=1e-9, ftol=1e-9)
            
            if result.success or result.cost < 1e-6:
                # Check if the solution is close to integers or half-integers
                u_len = U_shape[0] * U_shape[1]
                v_len = V_shape[0] * V_shape[1]
                w_len = W_shape[0] * W_shape[1]
                
                U_sol = result.x[:u_len].reshape(U_shape)
                V_sol = result.x[u_len:u_len+v_len].reshape(V_shape)
                W_sol = result.x[u_len+v_len:].reshape(W_shape)
                
                # Round to nearest half-integer
                U_int = np.round(U_sol * 2) / 2
                V_int = np.round(V_sol * 2) / 2
                W_int = np.round(W_sol * 2) / 2
                
                # Verify the integer solution
                T_check = np.zeros((n, m, p))
                for c in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            u_val = U_int[i*m + j, c]
                            if u_val == 0: continue
                            for k in range(p):
                                v_val = V_int[j*p + k, c]
                                if v_val == 0: continue
                                w_val = W_int[k*n + i, c]
                                T_check[i, j, k] += u_val * v_val * w_val
                                
                if np.allclose(T_check, 1, atol=1e-9):
                    # Found a valid rank 14 decomposition!
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
        except Exception as e:
            pass
            
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
