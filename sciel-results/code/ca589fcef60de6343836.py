"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x5x5.
    Uses the trivial decomposition which yields rank n*m*p = 50.
    This guarantees a valid exact decomposition.
    """
    n, m, p = DIMENSIONS
    
    # Trivial decomposition: one rank-1 term for each (i, j, k)
    total_rank = n * m * p
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i * m + j
                # B index: j * p + k
                # C index: k * n + i (as per task description W[k*n+i,r])
                
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
