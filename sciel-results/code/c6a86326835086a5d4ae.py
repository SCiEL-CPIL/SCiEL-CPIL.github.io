import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 4, 6
    
    # Target tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We want U (n*m, R), V (m*p, R), W (p*n, R)
    
    # Try R=71 first, then 72
    for R in [71, 72]:
        # Try multiple random initializations
        for attempt in range(100):
            np.random.seed(attempt)
            # Initialize U and V with small random values
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            
            # Function to compute W given U, V
            def compute_W(U, V):
                W = np.zeros((p*n, R))
                # For each k, i:
                # Solve A @ W_row = b
                # Where A is m x R, A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # b is m x 1, b[j] = 1
                for k in range(p):
                    for i in range(n):
                        # Construct A
                        A = np.zeros((m, R))
                        for j in range(m):
                            A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                        b = np.ones(m)
                        # Solve A @ w = b
                        # Use least squares
                        try:
                            w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                            W[k*n + i, :] = w
                        except:
                            W[k*n + i, :] = 0
                return W
            
            # Initial W
            W = compute_W(U, V)
            
            # Error function
            def error(params):
                # params are flattened U and V
                U_flat = params[:n*m*R]
                V_flat = params[n*m*R:]
                U_curr = U_flat.reshape(n*m, R)
                V_curr = V_flat.reshape(m*p, R)
                W_curr = compute_W(U_curr, V_curr)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                            val = np.dot(U_curr[i*m+j, :] * V_curr[j*p+k, :], W_curr[k*n+i, :])
                            err += (val - 1.0)**2
                return err
            
            # Optimize U and V
            x0 = np.concatenate([U.flatten(), V.flatten()])
            
            # Use L-BFGS-B for speed
            res = minimize(error, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.fun < 1e-9:
                # Solution found
                U_opt = res.x[:n*m*R].reshape(n*m, R)
                V_opt = res.x[n*m*R:].reshape(m*p, R)
                W_opt = compute_W(U_opt, V_opt)
                
                # Verify
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_opt[i*m+j, :] * V_opt[j*p+k, :], W_opt[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    # Convert to lists
                    return {
                        "U": U_opt.tolist(),
                        "V": V_opt.tolist(),
                        "W": W_opt.tolist()
                    }
    
    # Fallback to schoolbook
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
