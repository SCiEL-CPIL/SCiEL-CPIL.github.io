import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    def is_valid(U, V, W, R):
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0
                        for r in range(R):
                            s += U[i*m + j1, r] * V[j2*p + k, r] * W[k*n + i, r]
                        target = 1 if j1 == j2 else 0
                        if abs(s - target) > 1e-9:
                            return False
        return True

    def get_schoolbook():
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

    # Try structured search: fix U and V to small integer patterns, solve for W
    # Use the block structure insight: try to share factors across the 5 columns
    
    best_solution = get_schoolbook()
    best_rank = n * m * p

    # Generate candidate U, V matrices with entries in {-1, 0, 1}
    # Focus on patterns that might exploit the 3x3x3 + 3x3x2 block structure
    
    def try_rank_with_patterns(R, max_attempts=50):
        nonlocal best_solution, best_rank
        if R >= best_rank:
            return False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return False
            
            # Generate structured U and V
            # Pattern: use random {-1,0,1} but with some structure
            U = np.random.choice([-1, 0, 1], size=(n*m, R))
            V = np.random.choice([-1, 0, 1], size=(m*p, R))
            
            # Build linear system for W
            num_eq = n * p * m * m
            num_unk = p * n * R
            
            A = np.zeros((num_eq, num_unk))
            b = np.zeros(num_eq)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            for r in range(R):
                                coeff = U[i*m + j1, r] * V[j2*p + k, r]
                                if coeff != 0:
                                    w_idx = (k*n + i) * R + r
                                    A[eq_idx, w_idx] = coeff
                            b[eq_idx] = 1 if j1 == j2 else 0
                            eq_idx += 1
            
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if W is close to half-integers
            W_rounded = np.round(W_flat * 2) / 2
            W = W_rounded.reshape(p*n, R)
            
            if is_valid(U, V, W, R):
                best_solution = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = R
                return True
        
        return False

    # Try ranks from 44 down to 25
    for R in range(44, 24, -1):
        if time.time() > deadline - 15:
            break
        if try_rank_with_patterns(R, max_attempts=30):
            # Found a solution, try to go lower
            continue
    
    # If we found something better than schoolbook, return it
    if best_rank < n * m * p:
        return best_solution
    
    # Fallback to schoolbook
    return get_schoolbook()
