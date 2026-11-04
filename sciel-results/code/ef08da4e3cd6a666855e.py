import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook rank 72
    schoolbook_rank = n * m * p  # 72
    
    # Try to find a better decomposition using structured approach
    # Use Strassen on 2x2x2 subblocks where possible
    
    # For 3x4x6, let's try to use a block decomposition
    # Split m=4 into 2+2, p=6 into 3+3
    # This gives us four 3x2x3 sub-problems
    
    # Actually, let me try a more clever approach:
    # Use the fact that we can share some computations
    
    # For now, return schoolbook which is guaranteed correct
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]  # 12 x 72
    V = [[0] * rank for _ in range(m * p)]  # 24 x 72
    W = [[0] * rank for _ in range(p * n)]  # 18 x 72
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
