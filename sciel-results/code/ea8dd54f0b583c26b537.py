import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    n, m, p = 3, 3, 5
    
    # Schoolbook solution (Rank 45)
    best_rank = n * m * p
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1
                best_V[j*p+k, col] = 1
                best_W[k*n+i, col] = 1
                col += 1
                
    best_solution = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    
    # Discrete values for initialization
    disc_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Search for lower ranks from 44 down to 25
    for R in range(44, 24, -1):
        if time.time() > deadline - 15:
            break
            
        # Number of restarts depends on remaining time
        # Rough estimate: each restart takes ~0.5-1s?
        # Let's try up to 30 restarts if time permits
        max_restarts = 30
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(restart * 10000 + R * 100)
            
            # Initialize with discrete values
            U = np.random.choice(disc_vals, size=(n*m, R))
            V = np.random.choice(disc_vals, size=(m*p, R))
            W = np.random.choice(disc_vals, size=(p*n, R))
            
            # Add small noise to avoid symmetry breaking issues?
            # Or just use the discrete values as start.
            
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)
            
            # Precompute target
            target = np.ones((n, m, p))
            
            def objective(x):
                U_flat = x[:n*m*R].reshape(n, m, R)
                V_flat = x[n*m*R:n*m*R + m*p*R].reshape(m, p, R)
                W_flat = x[n*m*R + m*p*R:].reshape(p, n, R)
                
                # Compute S_ijk
                S = np.einsum('ijr,jkr,kir->ijk', U_flat, V_flat, W_flat)
                
                # Error is sum of (S - 1)^2
                diff = S - target
                error = np.sum(diff * diff)
                return error
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Define bounds
            bounds = [(-2.0, 2.0)] * len(x0)
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    # Verify exactness with rounded values
                    U_r_3d = U_rounded.reshape(n, m, R)
                    V_r_3d = V_rounded.reshape(m, p, R)
                    W_r_3d = W_rounded.reshape(p, n, R)
                    
                    S_check = np.einsum('ijr,jkr,kir->ijk', U_r_3d, V_r_3d, W_r_3d)
                    
                    if np.allclose(S_check, 1.0, atol=1e-9):
                        # Valid solution found
                        best_solution = {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        # Break restart loop to try lower rank
                        break
            except Exception:
                pass
                
    return best_solution
