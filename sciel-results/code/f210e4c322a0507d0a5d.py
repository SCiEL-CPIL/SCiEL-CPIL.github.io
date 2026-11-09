import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # Target tensor is all 1s
    # T[i, j, k] = 1
    
    # Try ranks from 62 down to 40
    # Start with a known good rank if possible, or just search
    
    best_result = None
    best_rank = n * m * p # 63
    
    # Schoolbook is always valid
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Search for lower rank
    # We will try to find integer/half-integer solutions
    # by optimizing floats and rounding
    
    for R in range(62, 39, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Use small integers to help convergence?
            # Or random floats?
            # Let's try random integers in {-1, 0, 1}
            U_init = np.random.randint(-1, 2, size=(n * m, R))
            V_init = np.random.randint(-1, 2, size=(m * p, R))
            W_init = np.random.randint(-1, 2, size=(p * n, R))
            
            # Flatten
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residuals(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Efficiently using einsum
                # U shape (n, m, R)
                # V shape (m, p, R)
                # W shape (p, n, R)
                U_3d = U.reshape(n, m, R)
                V_3d = V.reshape(m, p, R)
                W_3d = W.reshape(p, n, R)
                
                # Sum over r: U_{ijr} V_{jkr} W_{kir}
                # Note: W index is k*n + i, so W[k, i, r]
                # We want T_{ijk} = sum_r U_{ijr} V_{jkr} W_{kir}
                
                # Use einsum
                # 'ijr,jkr,kir->ijk'
                try:
                    prod = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                except Exception:
                    return np.ones(n*m*p)
                
                # Target is 1
                return (prod - 1).flatten()
            
            try:
                res = least_squares(residuals, x0, method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if res.success or np.linalg.norm(res.fun) < 1e-5:
                # Check if we can round to integers/half-integers
                U_opt = res.x[:n*m*R].reshape(n*m, R)
                V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_opt = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Round to nearest 0.5
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify exactness
                U_3d = U_round.reshape(n, m, R)
                V_3d = V_round.reshape(m, p, R)
                W_3d = W_round.reshape(p, n, R)
                
                try:
                    prod = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                except Exception:
                    continue
                
                if np.allclose(prod, 1, atol=1e-9):
                    # Convert to list of lists of ints/floats
                    U_list = [[int(x) if x == int(x) else x for x in row] for row in U_round]
                    V_list = [[int(x) if x == int(x) else x for x in row] for row in V_round]
                    W_list = [[int(x) if x == int(x) else x for x in row] for row in W_round]
                    
                    # Ensure they are numeric types compatible with return
                    # The prompt says "integer/half-integer entries"
                    # Python floats 0.5, 1.0 are fine.
                    
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = R
                    break # Found a solution for this rank, try lower rank? 
                    # Actually, we want minimal rank, so we should continue to lower R
                    # But we are iterating R downwards. 
                    # So we found a solution for R. We can break the attempt loop 
                    # and continue to R-1.
        
        # If we found a solution for R, we continue to R-1 to see if we can do better
        # If not, we continue.
        
    return best_result
