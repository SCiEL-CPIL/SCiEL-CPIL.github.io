"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as baseline
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
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to reduce rank via local search
    # We'll try removing terms and solving for adjustments
    U_curr = [row[:] for row in U]
    V_curr = [row[:] for row in V]
    W_curr = [row[:] for row in W]
    curr_rank = rank
    
    # Indices for the triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Try to remove each term and see if we can compensate
    for remove_idx in range(rank - 1, -1, -1):
        if time.time() > deadline - 20:
            break
        
        # Try removing term remove_idx
        test_U = [row[:] for row in U_curr]
        test_V = [row[:] for row in V_curr]
        test_W = [row[:] for row in W_curr]
        
        # Remove the term
        test_U = [row[:remove_idx] + row[remove_idx+1:] for row in test_U]
        test_V = [row[:remove_idx] + row[remove_idx+1:] for row in test_V]
        test_W = [row[:remove_idx] + row[remove_idx+1:] for row in test_W]
        test_rank = curr_rank - 1
        
        # Check if the remaining decomposition still works
        # This is expensive, so only do it if we have time
        if time.time() > deadline - 10:
            break
        
        valid = True
        for (i, j, k) in triples:
            s = 0.0
            for c in range(test_rank):
                s += test_U[i*m+j][c] * test_V[j*p+k][c] * test_W[k*n+i][c]
            if abs(s - 1.0) > 1e-9:
                valid = False
                break
        
        if valid:
            U_curr = test_U
            V_curr = test_V
            W_curr = test_W
            curr_rank = test_rank
            best_U, best_V, best_W = U_curr, V_curr, W_curr
            best_rank = curr_rank
    
    # Return the best found
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
