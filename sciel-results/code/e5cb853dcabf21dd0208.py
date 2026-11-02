import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Try to find lower rank decompositions
    # We search for ranks lower than schoolbook
    # Start from a reasonable guess, e.g., 55, down to 40
    # Or just iterate down from 59
    
    def is_nice(w_vec, tol=1e-6):
        # Check if w_vec is close to integers or half-integers
        # Multiply by 2 and check closeness to integers
        w2 = [2 * x for x in w_vec]
        for x in w2:
            if abs(x - round(x)) > tol:
                return False
        return True

    def solve_for_w(U, V, r):
        # U: nm x r
        # V: mp x r
        # Returns W: pn x r or None if no nice solution found
        W = [[0] * r for _ in range(p * n)]
        
        for i in range(n):
            for k in range(p):
                # System for W[k*n+i, :]
                # Equations for j in 0..m-1
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        A[j, c] = U[i * m + j][c] * V[j * p + k][c]
                
                # Solve A W_vec = 1
                # Use least squares if not square, or exact if possible
                # Since we want exact integer/half-integer, we check the solution
                try:
                    # Use numpy.linalg.lstsq
                    w_sol, residuals, rank, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    
                    # Check if solution is nice
                    if is_nice(w_sol):
                        # Convert to list of floats (integers or half-integers)
                        # Round to nearest half-integer
                        w_rounded = [round(x * 2) / 2.0 for x in w_sol]
                        for c in range(r):
                            W[k * n + i][c] = w_rounded[c]
                    else:
                        return None
                except Exception:
                    return None
        
        return W

    # Search strategy
    # Try ranks from 59 down to 40
    for r in range(59, 39, -1):
        if time.time() > deadline:
            break
        
        # Number of trials per rank
        # Increase trials for lower ranks? Or keep constant.
        trials = 500
        
        for _ in range(trials):
            if time.time() > deadline:
                break
            
            # Generate random sparse U and V
            # Density ~ 0.2
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            
            for i in range(n * m):
                for c in range(r):
                    if random.random() < 0.2:
                        U[i][c] = random.choice([-1, 1])
            
            for i in range(m * p):
                for c in range(r):
                    if random.random() < 0.2:
                        V[i][c] = random.choice([-1, 1])
            
            W = solve_for_w(U, V, r)
            
            if W is not None:
                # Verify the decomposition
                # Check sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(r):
                                s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                            if abs(s - 1) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = r
                    # We can break out of trials, but maybe try lower r?
                    # The loop goes down to 40, so it will continue to lower r.
                    # But we should update best_result.
                    pass

    return best_result
# EVOLVE-BLOCK-END
