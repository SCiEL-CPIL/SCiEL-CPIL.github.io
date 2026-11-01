"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p  # 48

    # The tensor for matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k]
    # We need to find matrices U (n*m x rank), V (m*p x rank), W (p*n, rank)
    # such that for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

    # For the schoolbook decomposition, we have rank = 48, and each equation
    # has exactly one non-zero term.

    # To find a lower rank, we need to solve a system of polynomial equations.
    # Given the time constraint, let's just return the schoolbook decomposition.

    rank = rank_school
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
