import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook decomposition
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Try to find a decomposition of lower rank
    # We will attempt ranks 71, 70, 69...
    # This is a heuristic, might not find it, but worth a try.
    
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    for R in range(rank_school - 1, 20, -1):
        if time.time() > deadline:
            break
            
        # Initialize with random values or schoolbook truncated?
        # Random initialization is better for avoiding local minima in non-convex problems?
        # But schoolbook is a valid point.
        # Let's try random initialization with small values.
        
        num_vars = (n * m + m * p + p * n) * R
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # Define the residual function
        # The tensor equation is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        # 0 otherwise (but indices are only valid i,j,k)
        
        def residuals(params):
            U_r = params[:n*m*R].reshape(n*m, R)
            V_r = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W_r = params[n*m*R + m*p*R:].reshape(p*n, R)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        val = np.sum(U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i])
                        res.append(val - 1.0)
            return res
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=10000)
            if result.success or result.cost < 1e-10:
                # Check if solution is close to integers/half-integers
                U_sol = result.x[:n*m*R].reshape(n*m, R)
                V_sol = result.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_sol = result.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Snap to nearest 0.5
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify exactness
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_snap[i*m+j] * V_snap[j*p+k] * W_snap[k*n+i])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                            
                if max_err < 1e-9:
                    return {
                        "U": to_list(U_snap),
                        "V": to_list(V_snap),
                        "W": to_list(W_snap)
                    }
        except Exception:
            pass
            
    # If no better rank found, return schoolbook
    return {
        "U": to_list(U_school),
        "V": to_list(V_school),
        "W": to_list(W_school)
    }
