"""Find rank-minimal tensor decomposition for 3x4 * 4x6 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct a tensor decomposition for matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' as lists of lists.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Time guard: if we ever add search, check deadline here
    elapsed = time.time() - start_time
    # For now, just return the schoolbook result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
