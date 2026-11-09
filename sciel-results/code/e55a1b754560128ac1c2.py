import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    def schoolbook():
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

    start_time = time.time()
    deadline = 120.0  # 120s limit, keep margin

    # Try to find a lower rank decomposition
    # Based on 3x3x3 (rank 23) + 3x3x2 (rank 18) = 41
    # We search for r in [41, 44]
    
    for r in range(41, 45):
        if time.time() > deadline - 20: # Stop if not enough time
            break
            
        # Initialize parameters
        # U: 9 x r, V: 15 x r, W: 15 x r
        # Total params: 9r + 15r + 15r = 39r
        # For r=41, 39*41 = 1599 params.
        
        np.random.seed(42)
        # Scale initialization to avoid huge numbers
        scale = 1.0
        U_init = np.random.randn(n*m, r) * scale
        V_init = np.random.randn(m*p, r) * scale
        W_init = np.random.randn(p*n, r) * scale
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def error_func(x):
            if time.time() > deadline - 5:
                return 1e10 # Timeout penalty
            
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W = x[n*m*r + m*p*r :].reshape(p*n, r)
            
            error = 0.0
            # Iterate over output indices i, k
            for i in range(n):
                for k in range(p):
                    # W index: k*n + i
                    w_row = W[k*n + i, :]  # shape (r,)
                    
                    # U indices: i*m + j for j in 0..m-1
                    u_block = U[i*m : i*m+m, :]  # shape (m, r)
                    
                    # V indices: j'*p + k for j' in 0..m-1
                    v_indices = [j*p + k for j in range(m)]
                    v_block = V[v_indices, :]  # shape (m, r)
                    
                    # A_t = w_row * u_block (element-wise)
                    # A has shape (m, r)
                    A = w_row[None, :] * u_block
                    
                    # B has shape (m, r)
                    B = v_block
                    
                    # M = A @ B.T
                    M = A @ B.T  # shape (m, m)
                    
                    # Target is Identity matrix of size m
                    # error += sum((M - I)^2)
                    I = np.eye(m)
                    error += np.sum((M - I)**2)
            
            return error

        # Use L-BFGS-B
        try:
            from scipy.optimize import minimize
            res = minimize(error_func, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5})
            
            if res.success or res.fun < 1e-6:
                # Check if we can rationalize
                x_sol = res.x
                U_sol = x_sol[:n*m*r].reshape(n*m, r)
                V_sol = x_sol[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_sol = x_sol[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Try to snap to half-integers
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify snapped solution
                x_snap = np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()])
                err_snap = error_func(x_snap)
                
                if err_snap < 1e-4:
                    # Convert to list of lists of floats
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                
        except Exception as e:
            pass
            
    # Fallback to schoolbook
    return schoolbook()
