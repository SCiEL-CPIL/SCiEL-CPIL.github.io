"""Rewritten program for matrix multiplication 3,4,5 tensor rank minimization."""
import numpy as np
import time
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
    
    # The target tensor for matrix multiplication A(nxm) * B(mx p) = C(nxp)
    # in the basis U[i,j], V[j,k], W[k,i] is 1 for all valid i,j,k.
    # T[i,j,k] = 1
    
    random.seed(42)
    
    # Try to find lower rank using random restarts
    # Focus on rank 59 first, as it's the most likely candidate for improvement if any.
    # If 59 is not found, 58 is even less likely.
    
    for r in range(rank_school - 1, max(nm, mp, pn) - 1, -1):
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
            
        # Try multiple random initializations
        # Use a mix of random Gaussian and sparse random to cover different structures
        num_trials = 20 if r > 50 else 5
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Sometimes solutions are close to schoolbook with one rank removed
            # Strategy: Take schoolbook, remove one random term, and optimize?
            # For now, use random Gaussian with small variance for stability
            
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1
            
            # Simple check: compute the resulting tensor and see error
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # We want T[i,j,k] = 1.0 for all i,j,k
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        target = 1.0
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
