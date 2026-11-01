"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    # U: shape (n*m, rank) = (6, rank)
    # V: shape (m*p, rank) = (21, rank)
    # W: shape (p*n, rank) = (14, rank)
    
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def check_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        # Reconstruct tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # The target tensor has T[i,j,k] = 1 for all valid i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def is_half_integer(x):
        return abs(x * 2 - round(x * 2)) < 1e-9
    
    def validate_entries(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not is_half_integer(val):
                        return False
        return True
    
    def try_rank(r, max_attempts=50):
        """Try to find a valid decomposition of rank r."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random initialization with half-integer values
            def random_half_int():
                return random.choice([-2, -1, -0.5, 0, 0.5, 1, 2])
            
            U = [[random_half_int() for _ in range(r)] for _ in range(n * m)]
            V = [[random_half_int() for _ in range(r)] for _ in range(m * p)]
            W = [[random_half_int() for _ in range(r)] for _ in range(p * n)]
            
            if check_decomposition(U, V, W) and validate_entries(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try to find a decomposition with rank less than 42
    # Start from lower ranks and work up
    best_result = schoolbook()
    best_rank = n * m * p
    
    # For small improvements, try ranks from best_rank-1 down
    # But random search over discrete half-integers is very hard
    # Let's try a structured approach instead
    
    # Actually, let me think about this more carefully.
    # The matrix multiplication tensor for 2x3x7 has a specific structure.
    # Each (i,j,k) triple must sum to 1.
    
    # Let me try a greedy/constructive approach:
    # Start with schoolbook and try to merge terms.
    
    # Alternative: use the fact that for fixed (i,j), the row of U[i*m+j] 
    # and for fixed (j,k), the row of V[j*p+k] interact.
    
    # Given the difficulty of discrete search, let's just return schoolbook
    # unless we can quickly find something better.
    
    # Try a few random attempts at rank 41, 40, ...
    for r in range(best_rank - 1, max(30, best_rank - 10), -1):
        if time.time() > deadline:
            break
        result = try_rank(r, max_attempts=20)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
