"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Strassen-based decomposition for 2x2 by 2x8 matrix multiplication.
    Splits the 2x8 matrix into 4 blocks of 2x2, applies Strassen to each.
    Rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Input A is 2x2: indices (i, j) for i in 0..1, j in 0..1
    # Input B is 2x8: indices (j, k) for j in 0..1, k in 0..7
    # Output C is 2x8: indices (i, k) for i in 0..1, k in 0..7
    # W convention: W[k*n + i, r]
    
    rank = 28  # 4 blocks * 7 Strassen multiplications
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 16 x 28
    
    # A indices: A[i*m + j] = A[i*2 + j], i in 0..1, j in 0..1
    # B indices: B[j*p + k] = B[j*8 + k], j in 0..1, k in 0..7
    # W indices: W[k*n + i] = W[k*2 + i], k in 0..7, i in 0..1
    
    # For each block b (0..3), the 2x2 block of B is columns 2b and 2b+1
    # B_block = [[B[0*8 + 2b], B[0*8 + 2b+1]],
    #            [B[1*8 + 2b], B[1*8 + 2b+1]]]
    # Let's name: e = B[2b], f = B[2b+1], g = B[8+2b], h = B[8+2b+1]
    
    # Strassen formulas for A = [[a, b], [c, d]]:
    # a = A[0], b = A[1], c = A[2], d = A[3]
    # 
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # Result block:
    # C[0, 2b]   = M1 + M4 - M5 + M7
    # C[0, 2b+1] = M3 + M5
    # C[1, 2b]   = M2 + M4
    # C[1, 2b+1] = M1 - M2 + M3 + M6
    
    for b in range(4):
        base_col = b * 7
        e_col = 2 * b       # B[0*8 + 2b]
        f_col = 2 * b + 1   # B[0*8 + 2b+1]
        g_col = 8 + 2 * b       # B[1*8 + 2b]
        h_col = 8 + 2 * b + 1   # B[1*8 + 2b+1]
        
        # M1 = (a+d)(e+h)
        col = base_col + 0
        U[0][col] = 1   # a
        U[3][col] = 1   # d
        V[e_col][col] = 1
        V[h_col][col] = 1
        # W: C[0,2b] += M1, C[1,2b+1] += M1
        W[(2*b) * n + 0][col] = 1
        W[(2*b+1) * n + 1][col] = 1
        
        # M2 = (c+d)e
        col = base_col + 1
        U[2][col] = 1   # c
        U[3][col] = 1   # d
        V[e_col][col] = 1
        # W: C[1,2b] += M2
        W[(2*b) * n + 1][col] = 1
        
        # M3 = a(f-h)
        col = base_col + 2
        U[0][col] = 1   # a
        V[f_col][col] = 1
        V[h_col][col] = -1
        # W: C[0,2b+1] += M3, C[1,2b+1] += M3
        W[(2*b+1) * n + 0][col] = 1
        W[(2*b+1) * n + 1][col] = 1
        
        # M4 = d(g-f)
        col = base_col + 3
        U[3][col] = 1   # d
        V[g_col][col] = 1
        V[f_col][col] = -1
        # W: C[0,2b] += M4, C[1,2b] += M4
        W[(2*b) * n + 0][col] = 1
        W[(2*b) * n + 1][col] = 1
        
        # M5 = (a+b)h
        col = base_col + 4
        U[0][col] = 1   # a
        U[1][col] = 1   # b
        V[h_col][col] = 1
        # W: C[0,2b] -= M5, C[0,2b+1] += M5
        W[(2*b) * n + 0][col] = -1
        W[(2*b+1) * n + 0][col] = 1
        
        # M6 = (c-a)(e+f)
        col = base_col + 5
        U[2][col] = 1   # c
        U[0][col] = -1  # -a
        V[e_col][col] = 1
        V[f_col][col] = 1
        # W: C[1,2b+1] += M6
        W[(2*b+1) * n + 1][col] = 1
        
        # M7 = (b-d)(g+h)
        col = base_col + 6
        U[1][col] = 1   # b
        U[3][col] = -1  # -d
        V[g_col][col] = 1
        V[h_col][col] = 1
        # W: C[0,2b] += M7
        W[(2*b) * n + 0][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
