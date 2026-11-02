"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 2 * p = 26 for n=2, m=2
    rank = 2 * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        r0 = 2 * k
        r1 = 2 * k + 1
        
        # U[i*m + j, r]
        # For r0 (associated with i=0): U[0*m+j, r0] = 1 for all j
        for j in range(m):
            U[0 * m + j][r0] = 1
            # U[1*m+j, r0] = 0 (already 0)
            
            # For r1 (associated with i=1): U[1*m+j, r1] = 1 for all j
            U[1 * m + j][r1] = 1
            # U[0*m+j, r1] = 0 (already 0)
            
            # V[j*p + k, r]
            # V is 1 for both r0 and r1 for this k
            V[j * p + k][r0] = 1
            V[j * p + k][r1] = 1
            
        # W[k*n + i, r]
        # For r0: W[k*n + 0, r0] = 1, W[k*n + 1, r0] = 0
        W[k * n + 0][r0] = 1
        # W[k * n + 1][r0] = 0
        
        # For r1: W[k*n + 0, r1] = 0, W[k*n + 1, r1] = 1
        # W[k * n + 0][r1] = 0
        W[k * n + 1][r1] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
