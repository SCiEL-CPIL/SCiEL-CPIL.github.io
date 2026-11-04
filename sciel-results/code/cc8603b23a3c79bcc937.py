import numpy as np
import random
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}

    # Strategy: Model the tensor decomposition as a linear programming problem over a finite field
    # We search for a sparse support pattern that admits an exact integer/half-integer solution
    
    start_time = time.time()
    max_time = 110  # seconds, leaving buffer for evaluation overhead
    best_rank = rank_schoolbook
    best_decomp = fallback
    
    # Try ranks from 35 to 62 (lower bound is theoretical min, upper is schoolbook)
    # For 3x3x7, known best is around 40-50
    for R in range(35, 63):
        if time.time() - start_time > max_time:
            break
        
        # Try several random sparse support patterns
        num_attempts = 50 if R < 50 else 20
        
        for attempt in range(num_attempts):
            if time.time() - start_time > max_time:
                break
            
            # Generate random sparse supports
            # Each rank-1 term has non-zero entries in U, V, W
            # We want to find a pattern where the system of equations has a solution
            
            U_support = np.zeros((n * m, R), dtype=int)
            V_support = np.zeros((m * p, R), dtype=int)
            W_support = np.zeros((p * n, R), dtype=int)
            
            # Randomly assign non-zero positions
            for c in range(R):
                i_u = random.randint(0, n * m - 1)
                i_v = random.randint(0, m * p - 1)
                i_w = random.randint(0, p * n - 1)
                U_support[i_u, c] = 1
                V_support[i_v, c] = 1
                W_support[i_w, c] = 1
            
            # Now we need to find coefficients u_c, v_c, w_c such that:
            # For all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is a system of n*m*p = 63 equations in 3*R unknowns
            # The equations are cubic (product of three variables), so it's nonlinear
            
            # Instead, let's try a different approach: assume the supports are such that
            # each term contributes to exactly one (i,j,k) triple, and the coefficient is 1
            # This is the schoolbook approach, but with shared terms
            
            # Better approach: Use the fact that matrix multiplication tensor has a specific structure
            # T[i,j,k] = 1 if the term (i,j,k) is present
            # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
            
            # Let's try to solve this using a greedy/iterative approach
            # Start with all coefficients zero, then try to satisfy each equation
            
            # Actually, let's use a simpler heuristic: 
            # For each rank-1 term, assign it to a random (i,j,k) triple
            # and set the coefficients to 1
            # Then check if the reconstruction is correct
            
            # Even simpler: Try to find a pattern where each (i,j,k) is covered exactly once
            # by some combination of terms, with coefficients that work out to 1
            
            # Let's try a structured approach based on the dimensions
            # For 3x3x7, we can try to exploit the rectangular shape
            
            # Generate a candidate decomposition using random weights
            U_vals = np.zeros((n * m, R))
            V_vals = np.zeros((m * p, R))
            W_vals = np.zeros((p * n, R))
            
            for c in range(R):
                # Randomly choose which (i,j) pair this term affects in U
                i_rand = random.randint(0, n - 1)
                j_rand = random.randint(0, m - 1)
                k_rand = random.randint(0, p - 1)
                
                # Set the support and random values
                U_vals[i_rand * m + j_rand, c] = random.choice([-1, 1])
                V_vals[j_rand * p + k_rand, c] = random.choice([-1, 1])
                W_vals[k_rand * n + i_rand, c] = random.choice([-1, 1])
            
            # Check if this decomposition works
            # Compute the reconstructed tensor
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(R):
                            val += U_vals[i * m + j, c] * V_vals[j * p + k, c] * W_vals[k * n + i, c]
                        error += abs(val - 1)
            
            if error < 1e-10:
                best_rank = R
                best_decomp = {
                    "U": U_vals.tolist(),
                    "V": V_vals.tolist(),
                    "W": W_vals.tolist()
                }
                break
        
        if best_rank < rank_schoolbook:
            break
    
    if best_rank < rank_schoolbook:
        return best_decomp
    return fallback
# EVOLVE-BLOCK-END
