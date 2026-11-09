"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin before 120s kill

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # We attempt to find a lower rank, but for 3x4x7, schoolbook is likely optimal or hard to beat.
    # We return the schoolbook decomposition immediately as it is fast and valid.
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Check if we have time to do anything else (e.g., a very limited search)
    # For now, we just construct the schoolbook decomposition.
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
