import numpy as np
import time
import scipy.optimize
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook solution
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Target: T_{ijk} = 1
    # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def residuals(params, R):
        # params is a flat vector of length 31*R
        U_flat = params[:R * n * m]
        V_flat = params[R * n * m : R * n * m + R * m * p]
        W_flat = params[R * n * m + R * m * p :]
        
        U_mat = U_flat.reshape(R, n * m).T # (n*m, R)
        V_mat = V_flat.reshape(R, m * p).T # (m*p, R)
        W_mat = W_flat.reshape(R, p * n).T # (p*n, R)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i * m + j, :], V_mat[j * p + k, :] * W_mat[k * n + i, :])
                    res.append(val - 1.0)
        return res

    # Try ranks from low to high
    # Lower bound is max(n*p, m*n, m*p) / min(n,m,p)? 
    # Let's start from 10
    
    for R in range(10, rank_school):
        if time.time() > deadline:
            break
            
        # Random initialization
        np.random.seed(42) # Fixed seed for reproducibility in search? Or random?
        # Let's try a few random restarts per R
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            init_params = np.random.randn(31 * R)
            
            try:
                sol = scipy.optimize.least_squares(residuals, init_params, args=(R,), method='lm')
                if sol.success and np.max(np.abs(sol.fun)) < 1e-5:
                    # Extract U, V, W
                    U_flat = sol.x[:R * n * m]
                    V_flat = sol.x[R * n * m : R * n * m + R * m * p]
                    W_flat = sol.x[R * n * m + R * m * p :]
                    
                    U_mat = U_flat.reshape(R, n * m).T
                    V_mat = V_flat.reshape(R, m * p).T
                    W_mat = W_flat.reshape(R, p * n).T
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U_mat * 2) / 2
                    V_rounded = np.round(V_mat * 2) / 2
                    W_rounded = np.round(W_mat * 2) / 2
                    
                    # Verify rounded solution
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_rounded[i * m + j, :], V_rounded[j * p + k, :] * W_rounded[k * n + i, :])
                                if abs(val - 1.0) > 1e-6:
                                    max_err = abs(val - 1.0)
                                    break
                            if max_err > 1e-6: break
                        if max_err > 1e-6: break
                    if max_err < 1e-6:
                        # Convert to list of lists
                        U_out = U_rounded.tolist()
                        V_out = V_rounded.tolist()
                        W_out = W_rounded.tolist()
                        return {"U": U_out, "V": V_out, "W": W_out}
            except Exception:
                pass
                
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
