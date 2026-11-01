"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Minimal rank is m * p = 30
    # Decomposition:
    # U[i*m + j, c] = 1 for all i, j, c
    # V[j*p + k, c] = 1 if c == j*p + k else 0 (Identity)
    # W[k*n + i, c] = 1 for all k, i, c
    # Sum over c: U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 * delta_{c, jp+k} * 1 = 1
    
    rank = m * p
    
    U = [[1] * rank for _ in range(n * m)]
    V = [[1 if i == j else 0 for j in range(rank)] for i in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
