import numpy as np
import random
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def verify_decomposition(U, V, W, rank):
        # Check if sum_c U[i,j,c] V[j,k,c] W[i,k,c] == 1 for all i,j,k
        # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Strategy: Search for low-rank decompositions using structured patterns
    # We'll try ranks from a lower bound up to schoolbook-1
    # For each rank, we try to find U, V, W with entries in {-1, 0, 1} or {-1, 0, 0.5, 1}
    
    start_time = time.time()
    time_limit = 115  # seconds
    
    # Lower bound for 3x4x5 is at least 20 (from known results)
    # Let's try ranks: 25, 26, 27, ..., 55
    # But to be efficient, we'll try specific ranks first
    
    # Known optimal rank for 3x4x5 is likely around 30-40
    # Let's try a range of ranks, starting from lower ones
    
    ranks_to_try = list(range(25, 56))
    
    # For each rank, try different coefficient sets
    # Coefficient sets: {-1, 0, 1}, {-1, 0, 0.5, 1}, {-1, -0.5, 0, 0.5, 1}
    
    coeff_sets = [
        [-1, 0, 1],
        [-1, -0.5, 0, 0.5, 1],
    ]
    
    for R in ranks_to_try:
        if R >= best_rank:
            continue
            
        if time.time() - start_time > time_limit:
            break
            
        # Try different coefficient sets
        for coeffs in coeff_sets:
            if time.time() - start_time > time_limit:
                break
                
            # Number of restarts per rank/coefficient set
            num_restarts = 100
            
            for r_idx in range(num_restarts):
                if time.time() - start_time > time_limit:
                    break
                    
                # Initialize U and V with random values from the coefficient set
                U_rand = [[random.choice(coeffs) for _ in range(R)] for _ in range(nm)]
                V_rand = [[random.choice(coeffs) for _ in range(R)] for _ in range(mp)]
                
                # Build the linear system for W
                # Variables: W[k*n+i, c]
                # Equations: For each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                
                A = np.zeros((n * m * p, R * n * p))
                b = np.ones(n * m * p)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Variable index for W is k * n + i
                            # For each c, the coefficient is U[i*m+j, c] * V[j*p+k, c]
                            # The column in A is (k*n+i) * R + c
                            for c in range(R):
                                coeff = U_rand[i * m + j][c] * V_rand[j * p + k][c]
                                w_idx = (k * n + i) * R + c
                                A[eq_idx, w_idx] = coeff
                            eq_idx += 1
                
                # Solve A W_flat = b
                try:
                    W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                except:
                    continue
                    
                # Reshape W_flat to (p*n, R)
                W_sol = W_flat.reshape((p * n, R))
                
                # Check if W_sol is close to the coefficient set
                # First, check if it's close to half-integers
                ok = True
                W_nice = []
                for row in W_sol:
                    res_row = []
                    for val in row:
                        # Find the closest value in coeffs
                        best_coeff = min(coeffs, key=lambda c: abs(val - c))
                        if abs(val - best_coeff) > 1e-6:
                            ok = False
                            break
                        # Convert to int if possible
                        if best_coeff == int(best_coeff):
                            res_row.append(int(best_coeff))
                        else:
                            res_row.append(best_coeff)
                    if not ok:
                        break
                    W_nice.append(res_row)
                
                if ok:
                    # Verify the decomposition with the rounded values
                    if verify_decomposition(U_rand, V_rand, W_nice, R):
                        best_rank = R
                        best_U = U_rand
                        best_V = V_rand
                        best_W = W_nice
                        return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
