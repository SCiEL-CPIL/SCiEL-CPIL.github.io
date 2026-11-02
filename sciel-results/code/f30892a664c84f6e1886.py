import numpy as np
import random
import time

DIMENSIONS = [2, 3, 3]

def check_decomposition(U, V, W, n, m, p, rank):
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    # Check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-6:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    
    # Try rank 15
    target_rank = 15
    start_time = time.time()
    time_limit = 10 # seconds
    
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
        # Random initialization
        U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p * n)]
        
        if check_decomposition(U, V, W, n, m, p, target_rank):
            return {"U": U, "V": V, "W": W}
            
    return {"U": U_s, "V": V_s, "W": W_s}
