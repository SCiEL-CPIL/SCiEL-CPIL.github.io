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
        """Verify that sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def is_valid_entries(matrix):
        """Check if all entries are integers or half-integers"""
        for row in matrix:
            for val in row:
                # Check if val is close to k/2 for some integer k
                nearest = round(val * 2) / 2
                if abs(val - nearest) > 1e-9:
                    return False
        return True
    
    def round_to_half_integers(matrix):
        """Round matrix entries to nearest half-integers"""
        result = []
        for row in matrix:
            res_row = []
            for val in row:
                nearest = round(val * 2) / 2
                if nearest == int(nearest):
                    res_row.append(int(nearest))
                else:
                    res_row.append(nearest)
            result.append(res_row)
        return result
    
    # Strategy: Sparse Random Search with Adaptive Restarts
    # Sparse matrices U and V are more likely to yield W with nice entries.
    
    start_time = time.time()
    time_limit = 115  # Safety margin
    
    # Search ranks from a lower bound up to schoolbook
    # Lower bound: max(nm, mp, pn) is not a strict lower bound for rank, but a good heuristic start.
    # Actually, rank >= max(n, m, p) is weak.
    # Let's start from 20.
    
    coeff_sets = [
        [-1, 0, 1],
        [-2, -1, 0, 1, 2]
    ]
    
    # Probability of non-zero entry in U and V
    sparsity_probs = [0.2, 0.3, 0.5]
    
    for R in range(20, min(61, rank_school + 1)):
        if time.time() - start_time > time_limit:
            break
        
        if R >= best_rank:
            continue
        
        # Adaptive restarts: more restarts for lower ranks
        if R < 30:
            num_restarts = 200
        elif R < 40:
            num_restarts = 100
        else:
            num_restarts = 50
            
        for coeff_set in coeff_sets:
            if time.time() - start_time > time_limit:
                break
                
            for sparsity in sparsity_probs:
                if time.time() - start_time > time_limit:
                    break
                    
                for r_idx in range(num_restarts):
                    if time.time() - start_time > time_limit:
                        break
                        
                    # Sparse Random Initialization
                    U_rand = [[0] * R for _ in range(nm)]
                    V_rand = [[0] * R for _ in range(mp)]
                    
                    for i in range(nm):
                        for c in range(R):
                            if random.random() < sparsity:
                                U_rand[i][c] = random.choice(coeff_set)
                    for j in range(mp):
                        for c in range(R):
                            if random.random() < sparsity:
                                V_rand[j][c] = random.choice(coeff_set)
                    
                    # Build the linear system for W
                    A = np.zeros((n * m * p, R * n * p))
                    b = np.ones(n * m * p)
                    
                    eq_idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
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
                    
                    W_sol = W_flat.reshape((p * n, R))
                    
                    # Check if W_sol is close to half-integers
                    if is_valid_entries(W_sol):
                        W_nice = round_to_half_integers(W_sol)
                        if verify_decomposition(U_rand, V_rand, W_nice, R):
                            best_rank = R
                            best_U = U_rand
                            best_V = V_rand
                            best_W = W_nice
                            return {"U": best_U, "V": best_V, "W": best_W}
    
    # If no lower rank found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
