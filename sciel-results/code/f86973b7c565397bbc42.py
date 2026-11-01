"""Matrix multiplication 2x3 x 3x9 rank decomposition."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # Conventions:
    #   U[i*m + j, c]
    #   V[j*p + k, c]
    #   W[k*n + i, c]
    # such that C[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
