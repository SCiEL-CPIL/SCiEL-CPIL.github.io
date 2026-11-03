"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Trivial rank-n*m*p decomposition
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find a lower-rank decomposition
    # Use a simple heuristic: try ranks from max(n*m, m*p, p*n) to rank-1
    min_possible = max(n*m, m*p, p*n)
    
    for try_rank in range(min_possible, rank):
        if time.time() > deadline:
            break
        
        # Try to find a rank-try_rank decomposition
        # Use random initialization and check if it works
        found = False
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            U_try = np.random.randn(nm, try_rank) * 0.1
            V_try = np.random.randn(mp, try_rank) * 0.1
            W_try = np.random.randn(pn, try_rank) * 0.1
            
            # Check if this decomposition is exact
            # T[i*m+j, j'*p+k, k'*n+i'] = sum_r U[i*m+j,r] * V[j'*p+k,r] * W[k'*n+i',r]
            # Should equal delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
            
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k in range(p):
                            for k2 in range(p):
                                for i2 in range(n):
                                    expected = 1 if (i==i2 and j==j2 and k==k2) else 0
                                    actual = np.sum(U_try[i*m+j, :] * V_try[j2*p+k, :] * W_try[k2*n+i2, :])
                                    err = abs(actual - expected)
                                    if err > max_err:
                                        max_err = err
                                    if max_err > 1e-6:
                                        break
                                if max_err > 1e-6:
                                    break
                            if max_err > 1e-6:
                                break
                        if max_err > 1e-6:
                            break
                    if max_err > 1e-6:
                        break
                if max_err > 1e-6:
                    break
            if max_err <= 1e-6:
                found = True
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = try_rank
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
