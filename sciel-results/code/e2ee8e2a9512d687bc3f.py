"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x3 matrix by a 3x10 matrix.
    
    The current implementation uses the schoolbook algorithm with rank 60.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] corresponds to A[i,j]
                U[i * m + j][column] = 1
                # V[j*p + k, column] corresponds to B[j,k]
                V[j * p + k][column] = 1
                # W[k*n + i, column] corresponds to C[i,k] (trace convention W[k*n+i])
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
