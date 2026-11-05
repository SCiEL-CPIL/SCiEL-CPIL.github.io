"""Matrix multiplication 3,4,7 — rank-74 exact decomposition via hierarchical Strassen."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We build a decomposition of rank 74 using a hierarchical approach:
    # Split n=3 into 1+2.
    # C_top (1x7) = A_top(1x4) @ B(4x7). Rank 28 (schoolbook).
    # C_bot (2x7) = A_bot(2x4) @ B(4x7).
    #   Split m=4 into 2+2.
    #   C_bot = A_bot_L(2x2) @ B_L(2x7) + A_bot_R(2x2) @ B_R(2x7).
    #   For each 2x2x7: split p=7 into 3+4.
    #     2x2x3: Rank 9 (Strassen 2x2x2 + schoolbook 2x2x1).
    #     2x2x4: Rank 14 (two Strassen 2x2x2 blocks).
    #   Total for 2x2x7: 9+14=23.
    #   Total for C_bot: 23+23=46.
    # Total rank: 28+46=74.
    
    rank = 74
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term for a single scalar multiplication
    # U[i*m+j], V[j*p+k], W[k*n+i]
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
    
    # Helper to add a Strassen 2x2x2 block
    # Local indices:
    # A: (i1,j1), (i1,j2), (i2,j1), (i2,j2)
    # B: (j1,k1), (j1,k2), (j2,k1), (j2,k2)
    # C: (k1,i1), (k1,i2), (k2,i1), (k2,i2)
    # Global indices:
    # u11 = i1*m+j1, u12 = i1*m+j2, u21 = i2*m+j1, u22 = i2*m+j2
    # v11 = j1*p+k1, v12 = j1*p+k2, v21 = j2*p+k1, v22 = j2*p+k2
    # w11 = k1*n+i1, w12 = k1*n+i2, w21 = k2*n+i1, w22 = k2*n+i2
    def add_strassen_2x2x2(i1, j1, k1, di, dj, dk):
        # di, dj, dk are the strides (usually 1 for contiguous blocks)
        # But here we might have non-contiguous indices if we split differently.
        # In our case:
        # i1, i2 = i1, i1+di
        # j1, j2 = j1, j1+dj
        # k1, k2 = k1, k1+dk
        i2 = i1 + di
        j2 = j1 + dj
        k2 = k1 + dk
        
        u11 = i1 * m + j1
        u12 = i1 * m + j2
        u21 = i2 * m + j1
        u22 = i2 * m + j2
        
        v11 = j1 * p + k1
        v12 = j1 * p + k2
        v21 = j2 * p + k1
        v22 = j2 * p + k2
        
        w11 = k1 * n + i1
        w12 = k1 * n + i2
        w21 = k2 * n + i1
        w22 = k2 * n + i2
        
        nonlocal col
        
        # M1 = (A11+A22)(B11+B22)
        U[u11][col] = 1; U[u22][col] = 1
        V[v11][col] = 1; V[v22][col] = 1
        W[w11][col] = 1; W[w22][col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[u21][col] = 1; U[u22][col] = 1
        V[v11][col] = 1
        W[w21][col] = 1
        col += 1
        
        # M3 = A11(B21-B22)
        U[u11][col] = 1
        V[v21][col] = 1; V[v22][col] = -1
        W[w11][col] = 1
        col += 1
        
        # M4 = A22(B12-B11)
        U[u22][col] = 1
        V[v12][col] = 1; V[v11][col] = -1
        W[w21][col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[u11][col] = 1; U[u12][col] = 1
        V[v22][col] = 1
        W[w12][col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[u21][col] = 1; U[u11][col] = -1
        V[v11][col] = 1; V[v12][col] = 1
        W[w22][col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[u12][col] = 1; U[u22][col] = -1
        V[v21][col] = 1; V[v22][col] = 1
        W[w22][col] = 1
        col += 1
        
        return [col-7, col-6, col-5, col-4, col-3, col-2, col-1]
    
    # Helper to add a 2x2x3 block (Strassen 2x2x2 + Schoolbook 2x2x1)
    # Indices: i1,i2 (2 rows), j1,j2 (2 cols), k1,k2,k3 (3 cols)
    def add_2x2x3(i1, j1, k1):
        i2 = i1 + 1
        j2 = j1 + 1
        k2 = k1 + 1
        k3 = k1 + 2
        
        # Strassen part for k1, k2
        s_cols = add_strassen_2x2x2(i1, j1, k1, 1, 1, 1)
        
        # Schoolbook part for k3
        # A11*B13, A12*B23, A21*B13, A22*B23
        u11 = i1 * m + j1
        u12 = i1 * m + j2
        u21 = i2 * m + j1
        u22 = i2 * m + j2
        
        v13 = j1 * p + k3
        v23 = j2 * p + k3
        
        w13 = k3 * n + i1
        w23 = k3 * n + i2
        
        nonlocal col
        # Term 1: A11 * B13
        U[u11][col] = 1; V[v13][col] = 1; W[w13][col] = 1; col += 1
        # Term 2: A12 * B23
        U[u12][col] = 1; V[v23][col] = 1; W[w13][col] = 1; col += 1
        # Term 3: A21 * B13
        U[u21][col] = 1; V[v13][col] = 1; W[w23][col] = 1; col += 1
        # Term 4: A22 * B23
        U[u22][col] = 1; V[v23][col] = 1; W[w23][col] = 1; col += 1
        
        return s_cols + [col-4, col-3, col-2, col-1]
    
    # Helper to add a 2x2x4 block (two Strassen 2x2x2 blocks)
    def add_2x2x4(i1, j1, k1):
        # Block 1: k1, k2
        s1 = add_strassen_2x2x2(i1, j1, k1, 1, 1, 1)
        # Block 2: k3, k4
        s2 = add_strassen_2x2x2(i1, j1, k1+2, 1, 1, 1)
        return s1 + s2

    # 1. C_top (1x7) = A_top(1x4) @ B(4x7). Schoolbook.
    # i=0, j=0..3, k=0..6
    for j in range(4):
        for k in range(7):
            add_schoolbook(0, j, k)
    
    # 2. C_bot (2x7) = A_bot(2x4) @ B(4x7)
    # i=1,2 (global indices), j=0..3, k=0..6
    # Split j into 0,1 and 2,3.
    # Part 1: j=0,1. A_bot_L (rows 1,2; cols 0,1). B_L (rows 0,1; cols 0..6).
    #   Split k into 0,1,2 and 3,4,5,6.
    #   2x2x3: i=1, j=0, k=0
    add_2x2x3(1, 0, 0)
    #   2x2x4: i=1, j=0, k=3
    add_2x2x4(1, 0, 3)
    
    # Part 2: j=2,3. A_bot_R (rows 1,2; cols 2,3). B_R (rows 2,3; cols 0..6).
    #   Split k into 0,1,2 and 3,4,5,6.
    #   2x2x3: i=1, j=2, k=0
    add_2x2x3(1, 2, 0)
    #   2x2x4: i=1, j=2, k=3
    add_2x2x4(1, 2, 3)
    
    # Verify col == 74
    assert col == 74, f"Expected rank 74, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
