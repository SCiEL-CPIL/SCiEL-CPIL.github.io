"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Use structured search with rational entries
    # For matrix multiplication, try to find low-rank decompositions
    # using entries from a small set of rationals
    
    import random
    import time
    
    start_time = time.time()
    time_limit = 115  # Leave margin for safety
    
    best_solution = None
    best_rank = None
    
    # Known lower bound: max(n*m, m*p, p*n) = 24
    # But we need to find actual achievable rank
    
    # Try structured approaches first
    # Approach 1: Try to use a decomposition based on polynomial multiplication
    # or other algebraic structures
    
    # Approach 2: Systematic search with small rational entries
    # Use values from {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
    
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    # For each candidate rank, try to find a solution
    # Start from a reasonable estimate
    
    # For 3x4 x 4x6, let's try ranks from 24 to 72
    # But use a smarter search: for each rank, do limited random trials
    
    for R in range(24, 73):
        if time.time() - start_time > time_limit:
            break
        
        # Number of trials per rank: more for lower ranks
        n_trials = max(10, 1000 - 10 * (R - 24))
        
        found = False
        for trial in range(n_trials):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization with structured bias
            # Bias towards sparse solutions
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            # For each column, decide which entries are non-zero
            # Try to make each column have few non-zeros
            
            for r in range(R):
                # Choose 1-3 non-zero entries for each of U[:,r], V[:,r], W[:,r]
                # U has n*m = 12 rows
                nz_u = random.randint(1, min(3, n*m))
                idx_u = random.sample(range(n*m), nz_u)
                for idx in idx_u:
                    U[idx, r] = random.choice(vals)
                
                # V has m*p = 24 rows
                nz_v = random.randint(1, min(3, m*p))
                idx_v = random.sample(range(m*p), nz_v)
                for idx in idx_v:
                    V[idx, r] = random.choice(vals)
                
                # W has p*n = 18 rows
                nz_w = random.randint(1, min(3, p*n))
                idx_w = random.sample(range(p*n), nz_w)
                for idx in idx_w:
                    W[idx, r] = random.choice(vals)
            
            # Check if this satisfies the tensor equations
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ui = U[i*m + j]
                        vj = V[j*p + k]
                        wk = W[k*n + i]
                        val = np.dot(ui * vj, wk)
                        if abs(val - 1) > 1e-9:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            
            if success:
                best_rank = R
                best_solution = (U, V, W)
                found = True
                break
        
        if found:
            break
    
    if best_solution is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    U, V, W = best_solution
    # Convert to lists of integers/half-integers
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
