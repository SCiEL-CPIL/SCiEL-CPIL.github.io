"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds to be safe
    
    # Generate the schoolbook solution as a fallback
    U_school = [[0]*rank_schoolbook for _ in range(n*m)]
    V_school = [[0]*rank_schoolbook for _ in range(m*p)]
    W_school = [[0]*rank_schoolbook for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find a lower rank decomposition using random search
    # For a given rank r, we need U (9xr), V (12xr), W (12xr) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # We'll try ranks from 35 down to 20
    for r in range(35, 19, -1):
        if time.time() > deadline:
            break
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize U, V, W with random small integers
            U = np.random.randint(-2, 3, size=(n*m, r))
            V = np.random.randint(-2, 3, size=(m*p, r))
            W = np.random.randint(-2, 3, size=(p*n, r))
            
            # Check if this gives a valid decomposition
            # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if best_U != U_school:
            break
    
    # If we found a better solution, return it; otherwise return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
