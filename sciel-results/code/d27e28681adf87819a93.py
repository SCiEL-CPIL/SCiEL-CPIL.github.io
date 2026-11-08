import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    # Try to find lower rank using scipy optimization
    # We try ranks from 47 down to 40 (just a few steps, as lower is very unlikely)
    for r in range(rank_sb - 1, 39, -1):
        if time.time() > deadline:
            break
        
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(100 + r + restart)
            
            # Flatten U, V, W into a single vector for optimization
            # Sizes: n*m*r, m*p*r, p*n*r
            size_u = n * m * r
            size_v = m * p * r
            size_w = p * n * r
            total_size = size_u + size_v + size_w
            
            x0 = np.random.randn(total_size)
            
            def objective(x):
                U = x[:size_u].reshape(n * m, r)
                V = x[size_u:size_u + size_v].reshape(m * p, r)
                W = x[size_u + size_v:].reshape(p * n, r)
                
                # Calculate error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            err += (s - 1.0) ** 2
                return err

            # Use L-BFGS-B
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'disp': False})
            
            if res.fun < 1e-6:
                # Snap to half-integers
                U_flat = res.x[:size_u]
                V_flat = res.x[size_u:size_u + size_v]
                W_flat = res.x[size_u + size_v:]
                
                U_snap = np.round(U_flat * 2) / 2
                V_snap = np.round(V_flat * 2) / 2
                W_snap = np.round(W_flat * 2) / 2
                
                # Verify
                U = U_snap.reshape(n * m, r)
                V = V_snap.reshape(m * p, r)
                W = W_snap.reshape(p * n, r)
                
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            max_err = max(max_err, abs(s - 1.0))
                
                if max_err < 1e-9:
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    best_rank = r
                    # If we found a lower rank, we can stop or continue to find even lower
                    # Since we iterate downwards, the first found is the lowest in this loop
                    return best_result

    return best_result
