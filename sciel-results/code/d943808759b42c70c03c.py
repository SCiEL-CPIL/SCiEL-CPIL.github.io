"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses the schoolbook decomposition with rank = n*m*p = 144.
    
    U: (n*m, rank) - A-space factors
    V: (m*p, rank) - B-space factors  
    W: (p*n, rank) - C-space factors (trace convention: W[k*n+i, r])
    
    The decomposition satisfies:
    For all (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    And zero for all other combinations.
    """
    deadline = time.time() + 120  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    rank = n * m * p  # 144
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 144
    V = [[0] * rank for _ in range(m * p)]  # 36 x 144
    W = [[0] * rank for _ in range(p * n)]  # 36 x 144
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A-space index: i*m + j
                # B-space index: j*p + k
                # C-space index: k*n + i (trace convention)
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
