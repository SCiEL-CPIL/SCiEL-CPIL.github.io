"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    n, m, p = 3, 3, 5
    # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (15, rank)
    # W shape: (p*n, rank) = (15, rank)
    # Convention: W[k*n+i, r]

    # Schoolbook decomposition: rank = n*m*p = 45
    # For each (i, j, k) triple, one rank-1 term
    rank = n * m * p  # 45

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
