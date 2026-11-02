import numpy as np
import random
import time

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

    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Try ranks from 20 up to 40
    # If we find a solution at R, we return it.
    ranks_to_try = list(range(20, 41))
    
    for R in ranks_to_try:
        if R >= best_rank:
            continue
            
        # Increase restarts for lower ranks? Or just a fixed number?
        # Let's do 1000 restarts per rank if time permits.
        num_restarts = 1000
        
        for r_idx in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U and V with random values from {-1, 0, 1}
            # To improve diversity, we can occasionally use {-2, -1, 0, 1, 2}
            # But let's stick to {-1, 0, 1} as it's the most likely for minimal rank
            # with "nice" coefficients.
            
            # Density of non-zeros: ~1/3 is good.
            U_rand = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(nm)]
            V_rand = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(mp)]
            
            # Build A and b
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
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = W_flat.reshape((p * n, R))
            
            ok, W_nice = is_half_integer(W_sol)
            if ok:
                if verify_decomposition(U_rand, V_rand, W_nice, R):
                    best_rank = R
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_nice
                    return {"U": best_U, "V": best_V, "W": best_W}

        if time.time() - start_time > time_limit:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
