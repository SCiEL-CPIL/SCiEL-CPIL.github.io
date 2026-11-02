import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Leave 30s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We'll build a decomposition of rank r
    # Part 1: 4x4x4 part (first 4 columns of B)
    # Part 2: 4x4x1 part (5th column of B)
    
    # For now, let's just use schoolbook for everything to get a valid solution
    # Then try to improve
    
    # Schoolbook: rank = n*m*p = 80
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
    
    # Verify
    # T[a,b,c] should equal sum_r U[a,r]*V[b,r]*W[c,r]
    # For a=i*m+j, b=j*p+k, c=k*n+i: should be 1
    # For other combos: should be 0
    
    return {"U": U, "V": V, "W": W}
