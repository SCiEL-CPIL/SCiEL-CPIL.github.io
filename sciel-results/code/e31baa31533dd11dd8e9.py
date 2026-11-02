import numpy as np
import random
import time
from fractions import Fraction

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
    
    start_time = time.time()
    time_limit = 110  # seconds
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def is_half_integer(matrix, tol=1e-9):
        for row in matrix:
            for val in row:
                nearest = round(val * 2) / 2
                if abs(val - nearest) > tol:
                    return False, None
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
        return True, result

    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def try_rank(R, num_trials=50):
        """Try to find a valid decomposition of rank R"""
        for trial in range(num_trials):
            if time.time() - start_time > time_limit:
                return None
            
            # Sparse random initialization with small integer entries
            # Use density that makes the system likely to be solvable
            density = 0.3
            U_rand = [[random.choice([-1, 0, 1]) if random.random() < density else 0 for _ in range(R)] for _ in range(nm)]
            V_rand = [[random.choice([-1, 0, 1]) if random.random() < density else 0 for _ in range(R)] for _ in range(mp)]
            
            # Build linear system for W
            # Equations: For each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Unknowns: W[k*n+i, c] for k in range(p), i in range(n), c in range(R)
            # Total unknowns: p * n * R
            # Total equations: n * m * p
            
            A = np.zeros((n * m * p, p * n * R))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(R):
                            coeff = U_rand[i * m + j][c] * V_rand[j * p + k][c]
                            if coeff != 0:
                                w_idx = (k * n + i) * R + c
                                A[eq_idx, w_idx] = coeff
                        eq_idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            W_sol = W_flat.reshape((p * n, R))
            
            # Check if close to half-integers
            ok, W_nice = is_half_integer(W_sol)
            if ok:
                if verify_decomposition(U_rand, V_rand, W_nice, R):
                    return {"U": U_rand, "V": V_rand, "W": W_nice}
        
        return None

    # Try ranks in increasing order from a reasonable lower bound
    # For 3x4x5, known lower bounds suggest rank >= 35
    # Try from 35 up to schoolbook-1
    for R in range(35, rank_school):
        if time.time() - start_time > time_limit:
            break
        
        result = try_rank(R, num_trials=100)
        if result is not None:
            return result
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
