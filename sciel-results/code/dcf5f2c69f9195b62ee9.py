"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Returns a dictionary with 'U', 'V', 'W' matrices.
    """
    n, m, p = DIMENSIONS
    rank = m * p  # 4 * 9 = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            # U part: picks out A[:, j]
            for i in range(n):
                U[i * m + j][col] = 1
            
            # V part: picks out B[j, :]
            for j_prime in range(m):
                for k_prime in range(p):
                    if j_prime == j and k_prime == k:
                        V[j_prime * p + k_prime][col] = 1
            
            # W part: places result in C[:, k]
            for k_prime in range(p):
                for i_prime in range(n):
                    if k_prime == k:
                        W[k_prime * n + i_prime][col] = 1
            
            col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
