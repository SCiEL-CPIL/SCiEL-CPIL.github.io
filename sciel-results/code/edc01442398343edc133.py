"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Uses symmetry-reduced tensor network approach exploiting n=2 structure.
    For each (j,k) pair, we need to handle two outputs (i=0,1). 
    We attempt to share rank-1 terms across the i dimension where the tensor structure allows.
    
    Strategy: For each (j,k), use 2 terms (one for i=0, one for i=1) = 40 total.
    But we can try to merge some by using linear combinations.
    
    Actually, let's try a different approach: exploit that for fixed j, the i=0 and i=1 
    computations are independent. The minimal rank for the full tensor is at least max(10,8,20)=20.
    
    Let's try rank 30 by merging some terms across j values.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Try to construct a rank-30 decomposition
    # For each k (5 values), we have n*m = 8 input pairs from A and m=4 inputs from B
    # Total terms needed per k: at least max(n, m) = 4? No, that's not right.
    
    # Let's use the schoolbook but try to find linear dependencies
    # For now, return schoolbook with rank 40
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
