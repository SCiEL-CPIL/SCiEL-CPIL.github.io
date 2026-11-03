"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-wise decomposition for 3x5 * 5x7 matrix multiplication.
    
    Strategy: Split the middle dimension (size 5) into blocks of size 3 and 2.
    - A (3x5) = [A1 (3x3) | A2 (3x2)]
    - B (5x7) = [B1 (3x7); B2 (2x7)]
    - C (3x7) = A1*B1 + A2*B2
    
    For each sub-problem, use the naive decomposition:
    - 3x3 * 3x7: rank = 3*3*7 = 63
    - 3x2 * 2x7: rank = 3*2*7 = 42
    Total rank = 63 + 42 = 105
    
    This matches the naive bound but is structured for potential future optimization.
    """
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # Split middle dimension into m1 + m2
    m1, m2 = 3, 2
    
    # Total rank for naive block-wise decomposition
    rank1 = n * m1 * p  # 3*3*7 = 63 for A1*B1
    rank2 = n * m2 * p  # 3*2*7 = 42 for A2*B2
    total_rank = rank1 + rank2  # 105
    
    # Initialize factor matrices
    U = [[0] * total_rank for _ in range(n * m)]   # shape (15, 105)
    V = [[0] * total_rank for _ in range(m * p)]   # shape (35, 105)
    W = [[0] * total_rank for _ in range(p * n)]   # shape (21, 105)
    
    # Process block 1: A1 (3x3) * B1 (3x7)
    # A1 occupies rows 0..2 of A, columns 0..2
    # B1 occupies rows 0..2 of B, all 7 columns
    # C contribution to all 3x7 output
    
    col = 0
    for i in range(n):        # i = 0,1,2 (rows of A1 and C)
        for j in range(m1):   # j = 0,1,2 (cols of A1 / rows of B1)
            for k in range(p):# k = 0..6 (cols of B1 and C)
                # A[i,j] index in U: i*m + j
                U[i * m + j][col] = 1
                # B[j,k] index in V: j*p + k
                V[j * p + k][col] = 1
                # C[i,k] index in W: k*n + i
                W[k * n + i][col] = 1
                col += 1
    
    # Process block 2: A2 (3x2) * B2 (2x7)
    # A2 occupies rows 0..2 of A, columns 3..4
    # B2 occupies rows 3..4 of B, all 7 columns
    # C contribution to all 3x7 output
    
    for i in range(n):        # i = 0,1,2 (rows of A2 and C)
        for j in range(m2):   # j = 0,1 (local cols of A2 / local rows of B2)
            j_global = m1 + j  # j = 3,4 (global index in B)
            for k in range(p):# k = 0..6 (cols of B2 and C)
                # A[i, j_global] index in U: i*m + j_global
                U[i * m + j_global][col] = 1
                # B[j_global, k] index in V: j_global*p + k
                V[j_global * p + k][col] = 1
                # C[i,k] index in W: k*n + i
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
