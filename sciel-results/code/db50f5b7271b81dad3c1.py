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
    
    start_time = time.time()
    time_limit = 110  # seconds
    
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

    def is_half_integer(matrix, tol=1e-6):
        """
        Checks if all values in matrix are close to half-integers.
        Returns (True, rounded_matrix) or (False, None).
        """
        rounded = []
        for row in matrix:
            res_row = []
            for val in row:
                nearest = round(val * 2) / 2.0
                if abs(val - nearest) > tol:
                    return False, None
                # Convert to int if possible for cleaner output, but float is fine too
                if nearest == int(nearest):
                    res_row.append(int(nearest))
                else:
                    res_row.append(nearest)
            rounded.append(res_row)
        return True, rounded

    # Search for lower rank
    # Known lower bounds suggest rank >= 30 for 3x4x5.
    # We try ranks from 30 upwards.
    # To find the *minimal* rank, we should iterate 30, 31, ...
    # But if 30 is impossible, we waste time.
    # Heuristic: Try 30, 35, 40, 45, 50. If we find one, we return it.
    # This might not be minimal, but it's an improvement over 60.
    # To be safer, we can try a few more ranks if time permits.
    
    ranks_to_try = [30, 32, 35, 38, 40, 42, 45, 48, 50]
    
    for R in ranks_to_try:
        if R >= best_rank:
            continue
            
        # Increase restarts for lower ranks as they are harder
        num_restarts = 500 if R < 35 else 200
        
        for r_idx in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U and V with random values from {-1, 0, 1}
            # Use sparsity to help the linear system be solvable with nice W
            # Density around 0.3 - 0.5
            density = 0.4
            U_rand = [[0] * R for _ in range(nm)]
            V_rand = [[0] * R for _ in range(mp)]
            
            for i in range(nm):
                for c in range(R):
                    if random.random() < density:
                        U_rand[i][c] = random.choice([-1, 1])
            
            for j in range(mp):
                for c in range(R):
                    if random.random() < density:
                        V_rand[j][c] = random.choice([-1, 1])
                        
            # Build the linear system for W
            # Variables: W[k*n+i, c]
            # Equations: For each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # A.shape = (n*m*p, R*n*p)
            
            A = np.zeros((n * m * p, R * n * p))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Variable index for W is k * n + i
                        # For each c, the coefficient is U[i*m+j, c] * V[j*p+k, c]
                        # The column in A is (k*n+i) * R + c
                        u_row = U_rand[i * m + j]
                        v_row = V_rand[j * p + k]
                        for c in range(R):
                            coeff = u_row[c] * v_row[c]
                            if coeff != 0:
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
            
            # Check if W_sol is close to half-integers
            ok, W_nice = is_half_integer(W_sol)
            if ok:
                # Verify the decomposition with the rounded values
                if verify_decomposition(U_rand, V_rand, W_nice, R):
                    best_rank = R
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_nice
                    # Found a valid one. Since we iterate ranks in increasing order,
                    # this is the minimal rank we found in this set.
                    # We can return immediately.
                    return {"U": best_U, "V": best_V, "W": best_W}
                    
        if time.time() - start_time > time_limit:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
