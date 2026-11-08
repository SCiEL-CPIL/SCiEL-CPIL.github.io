"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Target: Sum_r U[i,j,r] V[j',k,r] W[k,i,r] = delta_{j,j'}
    # For each (i,k), we need Sum_r W[k,i,r] (U[i,:,r] V[:,k,r]^T) = I_m
    
    best_result = None
    
    # Try ranks from 14 to 24
    for R in range(14, 25):
        if time.time() > deadline:
            break
        
        found = False
        # Try a few random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            np.random.seed(attempt)
            U_flat = np.random.uniform(-1, 1, (n * m * R))
            V_flat = np.random.uniform(-1, 1, (m * p * R))
            W_flat = np.random.uniform(-1, 1, (p * n * R))
            
            # Reshape
            U = U_flat.reshape(n, m, R)
            V = V_flat.reshape(m, p, R)
            W = W_flat.reshape(p, n, R)
            
            # Define residuals
            def residuals(params):
                U_p = params[0:n*m*R].reshape(n, m, R)
                V_p = params[n*m*R:n*m*R + m*p*R].reshape(m, p, R)
                W_p = params[n*m*R + m*p*R:].reshape(p, n, R)
                
                res = []
                for i in range(n):
                    for k in range(p):
                        # Matrix M = Sum_r W[k,i,r] * (U[i,:,r] * V[:,k,r].T)
                        M = np.zeros((m, m))
                        for r in range(R):
                            M += W_p[k, i, r] * np.outer(U_p[i, :, r], V_p[:, k, r])
                        # Target is Identity
                        res.extend((M - np.eye(m)).flatten())
                return np.array(res)
            
            x0 = np.concatenate([U_flat, V_flat, W_flat])
            
            try:
                # Use least squares
                result = least_squares(residuals, x0, method='lm', max_nfev=1000)
                if result.success or np.linalg.norm(result.fun) < 1e-6:
                    # Extract solution
                    U_sol = result.x[0:n*m*R].reshape(n, m, R)
                    V_sol = result.x[n*m*R:n*m*R + m*p*R].reshape(m, p, R)
                    W_sol = result.x[n*m*R + m*p*R:].reshape(p, n, R)
                    
                    # Round to half-integers
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    # Verify exactness
                    if _check_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
                        best_result = (R, U_int.tolist(), V_int.tolist(), W_int.tolist())
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    if best_result is None:
        # Fallback to schoolbook
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    R, U, V, W = best_result
    return {"U": U, "V": V, "W": W}

def _check_decomposition(U, V, W, n, m, p):
    R = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i * m + j][r] * V[jp * p + k][r] * W[k * n + i][r]
                    target = 1 if j == jp else 0
                    if not np.isclose(s, target):
                        return False
    return True
# EVOLVE-BLOCK-END
