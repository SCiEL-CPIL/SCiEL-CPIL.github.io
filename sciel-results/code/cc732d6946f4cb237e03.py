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
    time_limit = 110  # seconds
    
    # Search ranks from low to high to find the minimal rank.
    # Lower bound estimate: 20. Upper bound: 30 (likely exists).
    # We will try 20, 21, ..., 30.
    
    for R in range(20, 31):
        if R >= best_rank:
            continue
            
        # Try to find a solution for this R
        # We use random restarts.
        # To increase chance of success, we can vary the density of non-zeros in U and V.
        # But let's start with uniform random {-1, 0, 1}.
        
        # Time budget per rank: distribute remaining time.
        # Simple approach: try up to N restarts or until time limit.
        # Since lstsq is fast, we can do many.
        max_restarts = 2000
        
        for r_idx in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            U_rand = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(nm)]
            V_rand = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(mp)]
            
            # Build linear system A W = b
            # A is (n*m*p) x (R*n*p)
            # b is (n*m*p)
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
            
            # Check if W_sol is close to half-integers
            W_doubled = 2 * W_sol
            W_rounded = np.round(W_doubled)
            
            # Tolerance for half-integer check
            if np.max(np.abs(W_doubled - W_rounded)) < 1e-6:
                W_nice = (W_rounded / 2.0)
                
                # Verify exact decomposition
                if verify_decomposition(U_rand, V_rand, W_nice.tolist(), R):
                    best_rank = R
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_nice.tolist()
                    return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
