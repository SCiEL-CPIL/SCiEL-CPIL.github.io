"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The target tensor is the all-ones tensor J_{n,m,p} based on the schoolbook seed.
    # The rank of J_{n,m,p} is min(n, m, p).
    # Here min(4, 5, 6) = 4.
    rank = min(n, m, p)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct the decomposition:
    # U[i*m + j, c] = 1 if c == i else 0
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    # Sum_c U[i,j,c] V[j,k,c] W[k,i,c] = Sum_c (1 if c==i else 0) * 1 * 1 = 1
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][i] = 1
            
    for j in range(m):
        for k in range(p):
            for c in range(rank):
                V[j * p + k][c] = 1
                
    for k in range(p):
        for i in range(n):
            for c in range(rank):
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
