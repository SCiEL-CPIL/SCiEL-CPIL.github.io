import time
import numpy as np
import random

DIMENSIONS = [2, 3, 9]

def _validate(U, V, W, n=2, m=3, p=9, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                # For matrix multiplication, the tensor entry is 1 if the term is valid.
                # All terms are valid in dense MM.
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    t_start = time.time()
    DEADLINE = t_start + 115.0
    
    n, m, p = 2, 3, 9
    
    # Schoolbook
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
                
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    random.seed(12345)
    
    # Try ranks from lower bound to 53
    for r in range(27, 54):
        if time.time() > DEADLINE:
            break
            
        # Limit attempts per rank to avoid timeout
        max_attempts = 500 if r > 40 else 2000
        
        for attempt in range(max_attempts):
            if time.time() > DEADLINE:
                break
                
            # Random sparse matrices with small integers
            U = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(r)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(r)] for _ in range(p * n)]
            
            if _validate(U, V, W, n, m, p):
                best_rank = r
                best_result = {"U": U, "V": V, "W": W}
                break
                
    return best_result
