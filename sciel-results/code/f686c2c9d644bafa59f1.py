DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: Split m=5 into 2 and 3.
    # Part 1: 2x2x6 multiplication. Split p=6 into 2+2+2. Use Strassen for each 2x2x2.
    # Part 2: 2x3x6 multiplication. Split m=3 into 2 and 1.
    #   Part 2a: 2x2x6. Split p=6 into 2+2+2. Use Strassen.
    #   Part 2b: 2x1x6. Schoolbook.
    
    # We need to build U, V, W matrices.
    # Total rank = 3*7 (Part 1) + 3*7 (Part 2a) + 6*2 (Part 2b) = 21 + 21 + 12 = 54.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(i_offset, j_offset, k_offset, col_start):
        # Strassen for 2x2x2
        # A = [[a11, a12], [a21, a22]]
        # B = [[b11, b12], [b21, b22]]
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # Indices in U: i_offset + di, j_offset + dj
        # Indices in V: j_offset + dj, k_offset + dk
        # Indices in W: k_offset + dk, i_offset + di
        
        # M1
        U[i_offset + 0][col_start + 0] += 1
        U[i_offset + 1][col_start + 0] += 1
        V[j_offset + 0][col_start + 0] += 1
        V[j_offset + 1][col_start + 0] += 1
        W[k_offset + 0][col_start + 0] += 1
        W[k_offset + 1][col_start + 0] += 1
        
        # M2
        U[i_offset + 1][col_start + 1] += 1
        U[i_offset + 1][col_start + 1] += 1 # a21+a22
        V[j_offset + 0][col_start + 1] += 1 # b11
        W[k_offset + 0][col_start + 1] += 1
        W[k_offset + 1][col_start + 1] += 1
        
        # M3
        U[i_offset + 0][col_start + 2] += 1 # a11
        V[j_offset + 1][col_start + 2] += 1 # b12
        V[j_offset + 1][col_start + 2] += -1 # -b22? No, b12-b22.
        # Wait, V index is (j,k).
        # b12 is j=0, k=1. b22 is j=1, k=1.
        V[j_offset + 0][col_start + 2] += 1
        V[j_offset + 1][col_start + 2] += -1
        W[k_offset + 1][col_start + 2] += 1
        
        # M4
        U[i_offset + 1][col_start + 3] += 1 # a22
        V[j_offset + 1][col_start + 3] += 1 # b21
        V[j_offset + 0][col_start + 3] += -1 # -b11
        W[k_offset + 0][col_start + 3] += 1
        
        # M5
        U[i_offset + 0][col_start + 4] += 1 # a11
        U[i_offset + 0][col_start + 4] += 1 # a12
        V[j_offset + 1][col_start + 4] += 1 # b22
        W[k_offset + 1][col_start + 4] += 1
        
        # M6
        U[i_offset + 1][col_start + 5] += 1 # a21
        U[i_offset + 0][col_start + 5] += -1 # -a11
        V[j_offset + 0][col_start + 5] += 1 # b11
        V[j_offset + 0][col_start + 5] += 1 # b12
        W[k_offset + 1][col_start + 5] += 1
        
        # M7
        U[i_offset + 0][col_start + 6] += 1 # a12
        U[i_offset + 1][col_start + 6] += -1 # -a22
        V[j_offset + 1][col_start + 6] += 1 # b21
        V[j_offset + 1][col_start + 6] += 1 # b22
        W[k_offset + 1][col_start + 6] += 1
        
        return col_start + 7

    # Part 1: 2x2x6. i in 0..1, j in 0..1, k in 0..5.
    # Split k into 0-1, 2-3, 4-5.
    col = add_strassen_block(0, 0, 0, col)
    col = add_strassen_block(0, 0, 2, col)
    col = add_strassen_block(0, 0, 4, col)
    
    # Part 2: 2x3x6. i in 0..1, j in 2..4, k in 0..5.
    # Split j into 2-3 (2x2x6) and 4 (2x1x6).
    # Part 2a: 2x2x6. i in 0..1, j in 2..3, k in 0..5.
    col = add_strassen_block(0, 2, 0, col)
    col = add_strassen_block(0, 2, 2, col)
    col = add_strassen_block(0, 2, 4, col)
    
    # Part 2b: 2x1x6. i in 0..1, j in 4, k in 0..5.
    # Schoolbook: C_ik = A_i4 B_4k.
    for i in range(2):
        for k in range(6):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
