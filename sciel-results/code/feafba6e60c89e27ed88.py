import numpy as np
import random
import time
import itertools

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Schoolbook decomposition is optimal for M_{3,4,5} with rank 60.
    # We construct it directly to ensure speed and correctness.
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    best_rank = rank
    best_U = U
    best_V = V
    best_W = W

    # Helper: Check if a matrix is close to half-integers
    def is_half_integer(matrix, tol=1e-9):
        for row in matrix:
            for val in row:
                nearest = round(val * 2) / 2
                if abs(val - nearest) > tol:
                    return False, None
        # Convert to list of lists of float (or int if integer)
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

    # Helper: Verify decomposition
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

    # Strategy: Search for lower rank decompositions.
    # The theoretical lower bound for 3x4x5 is not 60, but finding a valid
    # integer/half-integer decomposition of lower rank is hard.
    # We attempt a structured search using random U, V and solving for W.
    
    # We will try ranks from a lower bound up to rank - 1.
    # A known lower bound is max(nm, mp, pn) = 20, but usually higher.
    # Let's try a range of ranks. If we find a valid one, we return it.
    
    # To maximize chance of success within time limit, we prioritize
    # ranks that are slightly below schoolbook, as they might be reachable
    # with sparse structures.
    
    ranks_to_try = [59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 48, 45, 40, 35, 30]
    
    for R in ranks_to_try:
        if R >= best_rank:
            continue
            
        if time.time() > deadline:
            break

        # Number of restarts per rank
        # Allocate time based on rank difficulty
        max_restarts = 50
        if R > 55:
            max_restarts = 100
        elif R > 50:
            max_restarts = 200
        else:
            max_restarts = 500

        for r_idx in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize U and V with random values from {-1, 0, 1}
            # Sparsity helps in finding structured solutions
            U_rand = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(nm)]
            V_rand = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(mp)]
            
            # Build the linear system for W
            # Variables: W[k*n+i, c]
            # Equations: For each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Note: W index is k*n+i.
            
            # A is sparse. We can use numpy lstsq.
            # A.shape = (60, R * 15)
            # b.shape = (60,)
            
            A = np.zeros((n * m * p, R * n * p))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Variable index for W is k * n + i
                        # For each c, the coefficient is U[i*m+j, c] * V[j*p+k, c]
                        # The column in A is (k*n+i) * R + c ? No.
                        # Let's flatten W as a vector of length R * n * p.
                        # Index in W vector: w_idx = (k*n+i) * R + c
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
                
            # Reshape W_flat to (n*p, R) then transpose to (p*n, R) to match interface?
            # Interface W shape: (p*n, R). Index k*n+i.
            # Our w_idx was (k*n+i)*R + c.
            # So W_flat is flattened in row-major order of (p*n, R).
            W_sol = W_flat.reshape((p * n, R))
            
            # Check if W_sol is close to half-integers
            ok, W_nice = is_half_integer(W_sol)
            if ok:
                # Verify the decomposition with the rounded values
                # Convert U_rand, V_rand to lists if needed (they are already lists)
                # W_nice is a list of lists
                if verify_decomposition(U_rand, V_rand, W_nice, R):
                    best_rank = R
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_nice
                    # Found a valid one. Since we iterate ranks in decreasing order
                    # of "difficulty" (or just a list), we can return immediately
                    # if we assume lower ranks are harder. 
                    # However, we want the MINIMAL rank. 
                    # If we find R=59, maybe R=58 exists? 
                    # The current loop tries specific ranks. If we find one, 
                    # it's a valid improvement. 
                    # To be safe and fast, we return the first valid one found 
                    # in the list of ranks we care about. 
                    # Given the time constraints, finding ANY lower rank is a win.
                    return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
