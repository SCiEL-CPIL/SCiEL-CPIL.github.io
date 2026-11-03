"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction
import random
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    deadline = time.time() + 110  # Leave 10s margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try to find lower rank using random restarts
    # For each rank r, try to find U, V, W such that the tensor equation holds
    # Use a simple iterative refinement
    
    random.seed(42)
    
    for r in range(rank_school - 1, max(nm, mp, pn) - 1, -1):
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
            
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
                
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Simple check: compute the resulting tensor and see error
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # We want T[i,j,k] = 1 if j==k else 0
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        target = 1.0 if j == k else 0.0
                        error += (val - target) ** 2
            
            if error < 1e-10:
                # Found a valid decomposition!
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
