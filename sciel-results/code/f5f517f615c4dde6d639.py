"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The seed program constructs the all-ones tensor of size n x m x p.
    # The rank of the all-ones tensor is min(n, m, p).
    rank = min(n, m, p)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U[i*m + j, r] = 1 if r == i (for i in 0..n-1)
    # V[j*p + k, r] = 1 for all
    # W[k*n + i, r] = 1 for all
    
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                if i == r:
                    U[i * m + j][r] = 1
        
        for j in range(m):
            for k in range(p):
                V[j * p + k][r] = 1
                
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
