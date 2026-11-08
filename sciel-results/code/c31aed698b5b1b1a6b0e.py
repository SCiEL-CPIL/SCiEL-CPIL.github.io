"""Matrix multiplication 2x5x6 tensor decomposition."""
import time

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # Rank 5 decomposition
    # The tensor for matrix multiplication C = AB can be decomposed as:
    # C_ik = sum_j A_ij B_jk
    # This is a sum of m rank-1 tensors.
    # Term j: U_j, V_j, W_j
    # U_j has 1s at (i, j) for all i.
    # V_j has 1s at (j, k) for all k.
    # W_j has 1s at (k, i) for all k, i.
    
    rank = m  # 5
    U = [[0] * rank for _ in range(n * m)]  # 10 x 5
    V = [[0] * rank for _ in range(m * p)]  # 30 x 5
    W = [[0] * rank for _ in range(p * n)]  # 12 x 5
    
    for j in range(m):
        col = j
        # Set U
        for i in range(n):
            U[i * m + j][col] = 1
        # Set V
        for k in range(p):
            V[j * p + k][col] = 1
        # Set W
        for k in range(p):
            for i in range(n):
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
