import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    target_rank = 15
    
    # Time management
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    def generate_target_tensor():
        # T[i, j, k] = 1 for all valid i, j, k
        # The condition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        # We flatten the indices for the solver
        # Indices:
        # U: (i, j, r) -> idx_u = (i*m + j)*R + r
        # V: (j, k, r) -> idx_v = (j*p + k)*R + r
        # W: (k, i, r) -> idx_w = (k*n + i)*R + r
        
        # We need to define the function f(params) -> residuals
        # params is a flat vector of size R*(n*m + m*p + p*n)
        
        pass

    # We will try to solve for rank 15
    R = target_rank
    
    # Total number of variables
    num_u = n * m * R
    num_v = m * p * R
    num_w = p * n * R
    total_vars = num_u + num_v + num_w
    
    # Precompute indices for the 18 equations
    # Equation for (i, j, k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] - 1 = 0
    # Note: U, V, W are flattened.
    # U[i,j,r] is at index (i*m + j)*R + r
    # V[j,k,r] is at index (j*p + k)*R + r
    # W[k,i,r] is at index (k*n + i)*R + r
    
    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx_base = (i * m + j) * R
                v_idx_base = (j * p + k) * R
                w_idx_base = (k * n + i) * R
                # For each r, we have a product of 3 variables
                # This is not linear.
                # least_squares minimizes sum of squares of residuals.
                # Residual is the value of the equation.
                
                eq_indices.append((u_idx_base, v_idx_base, w_idx_base))
    
    def residuals(params):
        # Split params
        U_flat = params[:num_u]
        V_flat = params[num_u:num_u+num_v]
        W_flat = params[num_u+num_v:]
        
        # Reshape to (dim1, dim2, R)
        U = U_flat.reshape((n*m, R))
        V = V_flat.reshape((m*p, R))
        W = W_flat.reshape((p*n, R))
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    u_row = U[i*m+j]
                    v_row = V[j*p+k]
                    w_row = W[k*n+i]
                    val = np.sum(u_row * v_row * w_row)
                    res[idx] = val - 1.0
                    idx += 1
        return res

    # Initial guess: Random
    # Or better: Use the rank 18 solution structure?
    # Rank 18 has 18 terms. We are using 15.
    # Random initialization is safer to avoid local minima of the "zero" solution.
    
    best_solution = None
    best_error = np.inf
    
    # Try a few random restarts
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        # Scale: values around 1
        x0 = np.random.randn(total_vars)
        
        # Run optimization
        try:
            result = opt.least_squares(residuals, x0, method='lm')
            final_error = result.cost
            if final_error < best_error:
                best_error = final_error
                best_solution = result.x
        except Exception:
            pass

    # Check if we found a good solution
    if best_solution is not None and best_error < 1e-6:
        # Reconstruct U, V, W
        U_flat = best_solution[:num_u]
        V_flat = best_solution[num_u:num_u+num_v]
        W_flat = best_solution[num_u+num_v:]
        
        U = U_flat.reshape((n*m, R))
        V = V_flat.reshape((m*p, R))
        W = W_flat.reshape((p*n, R))
        
        # Try to round to integers or half-integers
        # The problem states "integer/half-integer entries"
        # We can round to nearest 0.5
        
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        # Verify the rounded solution
        # Check if the rounded solution satisfies the equations
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_int[i*m+j] * V_int[j*p+k] * W_int[k*n+i])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Return as list of lists
            return {
                "U": U_int.tolist(),
                "V": V_int.tolist(),
                "W": W_int.tolist()
            }

    # Fallback to rank 18
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
