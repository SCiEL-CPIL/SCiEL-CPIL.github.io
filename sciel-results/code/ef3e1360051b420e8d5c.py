"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the problem into 5 blocks of 2x3x2
    # Each block has rank 11
    # Total rank 55
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 5 blocks of 2 columns in B
    for block in range(5):
        # B columns for this block are 2*block and 2*block+1
        b_col_0 = 2 * block
        b_col_1 = 2 * block + 1
        
        # Indices in V (m*p = 30)
        # V row j*p + k
        # j is row of B (0,1,2), k is col of B (0..9)
        
        # Term 0: (c+f)z -> C_00 += cz, C_10 += fz
        # c is A_02 (U[2]), f is A_12 (U[5])
        # z is B_20 (V[2*10 + b_col_0])
        # C_00 is W[0*n + 0] = W[0]
        # C_10 is W[1*n + 0] = W[2]
        
        U[2][col] = 1  # c
        U[5][col] = 1  # f
        V[2 * 10 + b_col_0][col] = 1  # z
        W[0][col] = 1  # C_00
        W[2][col] = 1  # C_10
        col += 1
        
        # Term 1: c w -> C_01 += cw
        # w is B_21 (V[2*10 + b_col_1])
        # C_01 is W[0*n + 1] = W[1]
        
        U[2][col] = 1  # c
        V[2 * 10 + b_col_1][col] = 1  # w
        W[1][col] = 1  # C_01
        col += 1
        
        # Term 2: f w -> C_11 += fw
        # C_11 is W[1*n + 1] = W[3]
        
        U[5][col] = 1  # f
        V[2 * 10 + b_col_1][col] = 1  # w
        W[3][col] = 1  # C_11
        col += 1
        
        # Term 3: a x -> C_00 += ax
        # a is A_00 (U[0]), x is B_00 (V[0*10 + b_col_0])
        
        U[0][col] = 1
        V[0 * 10 + b_col_0][col] = 1
        W[0][col] = 1
        col += 1
        
        # Term 4: b y -> C_00 += by
        # b is A_01 (U[1]), y is B_10 (V[1*10 + b_col_0])
        
        U[1][col] = 1
        V[1 * 10 + b_col_0][col] = 1
        W[0][col] = 1
        col += 1
        
        # Term 5: a u -> C_01 += au
        # u is B_01 (V[0*10 + b_col_1])
        
        U[0][col] = 1
        V[0 * 10 + b_col_1][col] = 1
        W[1][col] = 1
        col += 1
        
        # Term 6: b v -> C_01 += bv
        # v is B_11 (V[1*10 + b_col_1])
        
        U[1][col] = 1
        V[1 * 10 + b_col_1][col] = 1
        W[1][col] = 1
        col += 1
        
        # Term 7: d x -> C_10 += dx
        # d is A_10 (U[3]), x is B_00
        
        U[3][col] = 1
        V[0 * 10 + b_col_0][col] = 1
        W[2][col] = 1
        col += 1
        
        # Term 8: e y -> C_10 += ey
        # e is A_11 (U[4]), y is B_10
        
        U[4][col] = 1
        V[1 * 10 + b_col_0][col] = 1
        W[2][col] = 1
        col += 1
        
        # Term 9: d u -> C_11 += du
        # u is B_01
        
        U[3][col] = 1
        V[0 * 10 + b_col_1][col] = 1
        W[3][col] = 1
        col += 1
        
        # Term 10: e v -> C_11 += ev
        # v is B_11
        
        U[4][col] = 1
        V[1 * 10 + b_col_1][col] = 1
        W[3][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
