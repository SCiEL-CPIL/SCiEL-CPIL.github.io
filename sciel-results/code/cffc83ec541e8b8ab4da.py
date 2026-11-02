import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 15 decomposition: compute each of the n rows of C independently.
    # Row i of C is A[i, :] * B.
    # This is a 1 x m x p multiplication, which has rank m.
    # Total rank is n * m.
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        # For each row i of A, we perform a 1 x m x p multiplication.
        # The rank-m decomposition for 1 x m x p is:
        # C_k = sum_j A_j B_{jk}
        # T = sum_j e_j (in A) otimes e_{jk} (in B) otimes e_k (in C)
        # This is a sum of m rank-1 tensors.
        for j in range(m):
            # U component: A[i, j]
            U[i * m + j][col] = 1
            # V component: B[j, :]
            for k in range(p):
                V[j * p + k][col] = 1
            # W component: C[i, :]
            for k in range(p):
                W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
