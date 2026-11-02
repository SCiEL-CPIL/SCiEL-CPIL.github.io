"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition of rank 28.
    # C = A1*B1 + A2*B2, where A=[A1, A2] and B=[B1; B2].
    # A1, A2 are 2x2. B1, B2 are 2x4.
    # A1*B1 = A1*B11 + A1*B12 (two 2x2x2 multiplications)
    # A2*B2 = A2*B21 + A2*B22 (two 2x2x2 multiplications)
    # Total 4 * 7 = 28 rank-1 terms.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen M222 block
    # a, b, c, d are indices in A (flattened row-major)
    # e, f, g, h are indices in B (flattened row-major)
    # i, j, k, l are indices in C (flattened row-major)
    # col is the current column in U, V, W
    
    def add_strassen_block(a_idx, b_idx, c_idx, d_idx, 
                           e_idx, f_idx, g_idx, h_idx,
                           c00, c01, c10, c11, col):
        # p1 = (a+d)(e+h)
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # p2 = (c+d)e
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c01][col] += 1
        col += 1
        
        # p3 = a(f-h)
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] += -1
        W[c10][col] += 1
        col += 1
        
        # p4 = d(g-f)
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[f_idx][col] += -1
        W[c00][col] += 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # p5 = (a+b)h
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00][col] += -1
        W[c10][col] += 1
        col += 1
        
        # p6 = (c-d)(e+f)
        U[c_idx][col] += 1
        U[d_idx][col] += -1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11][col] += 1
        col += 1
        
        # p7 = (b-c)(g+h)
        U[b_idx][col] += 1
        U[c_idx][col] += -1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00][col] += 1
        W[c11][col] += -1
        col += 1
        
        return col

    # Indices in A (2x4):
    # A1 is top-left 2x2:
    # a00=0, a01=1, a10=2, a11=3
    # A2 is top-right 2x2:
    # a02=4, a03=5, a12=6, a13=7
    
    # Indices in B (4x4):
    # B1 is top 2x4:
    # b00=0, b01=1, b02=2, b03=3
    # b10=4, b11=5, b12=6, b13=7
    # B2 is bottom 2x4:
    # b20=8, b21=9, b22=10, b23=11
    # b30=12, b31=13, b32=14, b33=15
    
    # Indices in C (4x2) -> flattened p*n = 4*2=8?
    # Wait, C is 2x4. p=4, n=2.
    # W shape is (p*n, rank) = (8, rank).
    # W[k*n+i, r].
    # C[i, j] corresponds to W[j*n + i].
    # C00 -> W[0*2+0]=0
    # C01 -> W[1*2+0]=2
    # C02 -> W[2*2+0]=4
    # C03 -> W[3*2+0]=6
    # C10 -> W[0*2+1]=1
    # C11 -> W[1*2+1]=3
    # C12 -> W[2*2+1]=5
    # C13 -> W[3*2+1]=7
    
    # Block 1: A1 * B11 (B11 is top-left 2x2 of B)
    # A1 indices: 0,1,2,3
    # B11 indices: 0,1,4,5
    # C part: C00, C01, C10, C11 -> W[0], W[2], W[1], W[3]
    col = add_strassen_block(0, 1, 2, 3, 0, 1, 4, 5, 0, 2, 1, 3, 0)
    
    # Block 2: A1 * B12 (B12 is top-right 2x2 of B)
    # A1 indices: 0,1,2,3
    # B12 indices: 2,3,6,7
    # C part: C02, C03, C12, C13 -> W[4], W[6], W[5], W[7]
    col = add_strassen_block(0, 1, 2, 3, 2, 3, 6, 7, 4, 6, 5, 7, col)
    
    # Block 3: A2 * B21 (B21 is bottom-left 2x2 of B)
    # A2 indices: 4,5,6,7
    # B21 indices: 8,9,12,13
    # C part: C00, C01, C10, C11 -> W[0], W[2], W[1], W[3]
    col = add_strassen_block(4, 5, 6, 7, 8, 9, 12, 13, 0, 2, 1, 3, col)
    
    # Block 4: A2 * B22 (B22 is bottom-right 2x2 of B)
    # A2 indices: 4,5,6,7
    # B22 indices: 10,11,14,15
    # C part: C02, C03, C12, C13 -> W[4], W[6], W[5], W[7]
    col = add_strassen_block(4, 5, 6, 7, 10, 11, 14, 15, 4, 6, 5, 7, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
