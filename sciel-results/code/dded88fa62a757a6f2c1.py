"""Matrix multiplication 3,5,7 - rank-minimal tensor decomposition search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == delta_{j,k}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                expected = 1.0 if j == k else 0.0
                if abs(s - expected) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, n*p, m*p)
    lower_bound = max(n*m, n*p, m*p)
    
    # Search ranks from lower_bound upward
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random integer/half-integer matrices
            scale = 2  # Use half-integers: values are multiples of 0.5
            U = [[random.randint(-scale, scale) / 2.0 for _ in range(rank)] for _ in range(n*m)]
            V = [[random.randint(-scale, scale) / 2.0 for _ in range(rank)] for _ in range(m*p)]
            W = [[random.randint(-scale, scale) / 2.0 for _ in range(rank)] for _ in range(p*n)]
            
            if verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                break
    
    return best_result
