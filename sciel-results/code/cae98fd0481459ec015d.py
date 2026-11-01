"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # The condition for the decomposition is:
    # For all i in [0, n), j in [0, m), k in [0, p):
    #   sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1
    
    # Schoolbook decomposition has rank = n * m * p
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
    
    # Verify the decomposition (optional, for debugging)
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             s = 0
    #             for r in range(rank):
    #                 s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    #             if s != 1:
    #                 raise ValueError("Decomposition failed")

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
