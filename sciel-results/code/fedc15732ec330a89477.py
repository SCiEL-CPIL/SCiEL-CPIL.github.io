"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130  # 130 seconds to be safe
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # The multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # Try ranks from 55 down to 60
    for target_rank in range(55, 61):
        if time.time() > deadline - 30:
            break
        
        # Try random initializations and solve
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random support pattern
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Initialize with random ±1 or ±0.5 entries
            for c in range(target_rank):
                for idx in range(nm):
                    if np.random.random() < 0.3:
                        U[idx, c] = np.random.choice([-1, 1, -0.5, 0.5])
                for idx in range(mp):
                    if np.random.random() < 0.3:
                        V[idx, c] = np.random.choice([-1, 1, -0.5, 0.5])
                for idx in range(pn):
                    if np.random.random() < 0.3:
                        W[idx, c] = np.random.choice([-1, 1, -0.5, 0.5])
            
            # Check if this gives the correct tensor
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Convert to lists
                best = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                break
        
        if best_rank < rank_school:
            break
    
    return best
# EVOLVE-BLOCK-END
