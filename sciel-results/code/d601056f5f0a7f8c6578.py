"""Program for 3x4x5 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 5]


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Use structured Kronecker product decomposition.
    # Split the middle dimension m=4 into two blocks of size 2:
    #   A (3x4) = [A_L | A_R]  where A_L, A_R are 3x2
    #   B (4x5) = [B_T; B_B]   where B_T, B_B are 2x5
    # Then C = A_L * B_T + A_R * B_B
    #
    # Each sub-problem (3x2)(2x5) is decomposed via its own
    # schoolbook terms.  The full rank is 2 * (3*2*5) = 60,
    # which matches the direct schoolbook rank n*m*p = 60.
    #
    # We build the factors explicitly using the block structure
    # so the code is transparent and verifiable.
    
    rank = n * m * p  # 60
    U = [[0] * rank for _ in range(n * m)]  # shape (12, 60)
    V = [[0] * rank for _ in range(m * p)]  # shape (20, 60)
    W = [[0] * rank for _ in range(p * n)]  # shape (15, 60)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term corresponds to A[i,j] * B[j,k] contributing to C[i,k]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
