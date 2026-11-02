"""Matrix multiplication 2x2x4 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 4
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    rank = 14
    
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # We use a "middle product" decomposition:
    # The 2x2x4 multiplication can be viewed as two 2x2x2 multiplications
    # (for columns 0-1 and 2-3 of B) plus cross-terms that exploit the
    # rectangular structure.
    #
    # For a standard Strassen 2x2x2, we have 7 terms. For 2x2x4, we need
    # to handle 4 output columns. The naive approach gives 14 terms (2 blocks
    # of 7). But we can do better by sharing some terms across blocks.
    #
    # Here's the key insight: for the 2x2x4 case, we can use 14 terms where
    # the V-factors for the second block are linear combinations that share
    # support with the first block, reducing the effective rank.
    #
    # Let's define the 14 terms explicitly. Each term has:
    #   u_entries: list of (i, j, val) for the U factor (4 x rank)
    #   v_entries: list of (j, k, val) for the V factor (8 x rank)  
    #   w_entries: list of (k, i, val) for the W factor (8 x rank)
    #
    # The output C[i,k] = sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where the sum over r gives the correct product.
    
    # Term 0: M0 = (a00+a11)(b00+b20)  -- combines first column of both blocks
    add_term_0 = True
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[4][0] += 1  # b00 (j=0,k=0) and b20 (j=0,k=2)
    W[0][0] += 1; W[4][1] += 1  # c00 (k=0,i=0) and c21 (k=2,i=1)
    
    # Term 1: M1 = (a10+a11)(b01+b21)
    U[2][1] += 1; U[3][1] += 1
    V[1][1] += 1; V[5][1] += 1  # b01 (j=0,k=1) and b21 (j=0,k=3)
    W[1][1] += 1; W[5][0] += -1  # c01 (k=1,i=0) coeff 1, c20 (k=2,i=0) coeff -1
    
    # Term 2: M2 = a00(b10+b30)
    U[0][2] += 1
    V[2][2] += 1; V[6][2] += 1  # b10 (j=1,k=0) and b30 (j=1,k=2)
    W[0][2] += 1; W[4][2] += 1  # c00 (k=0,i=0) and c20 (k=2,i=0)
    
    # Term 3: M3 = a11(b11+b31)
    U[3][3] += 1
    V[3][3] += 1; V[7][3] += 1  # b11 (j=1,k=1) and b31 (j=1,k=3)
    W[1][3] += 1; W[5][3] += 1  # c01 (k=1,i=1) and c21 (k=3,i=1)
    
    # Term 4: M4 = (a00+a01)(b20+b21)
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1; V[5][4] += 1  # b20 (j=0,k=2) and b21 (j=0,k=3)
    W[4][4] += 1; W[5][4] += -1  # c20 (k=2,i=0) and c21 (k=3,i=0)
    
    # Term 5: M5 = (a10-a00)(b21-b20)
    U[2][5] += 1; U[0][5] += -1
    V[5][5] += 1; V[4][5] += -1  # b21 (j=0,k=3) and b20 (j=0,k=2)
    W[5][5] += 1  # c21 (k=3,i=1)
    
    # Term 6: M6 = (a01-a11)(b30+b31)
    U[1][6] += 1; U[3][6] += -1
    V[6][6] += 1; V[7][6] += 1  # b30 (j=1,k=2) and b31 (j=1,k=3)
    W[4][6] += 1  # c20 (k=2,i=0)
    
    # Term 7: M7 = (a10+a11)(b10+b30)  -- wait, this is getting complex
    # Let me restart with a cleaner approach.
    
    # Actually, let's use the known rank-14 decomposition for 2x2x4.
    # The standard approach: use Strassen's 7 terms for each of the two
    # 2x2x2 blocks, giving 14 terms. This is already optimal for this
    # size.
    
    # Reset and use clean Strassen-based construction
    for row in range(u_dim):
        for col in range(rank):
            U[row][col] = 0
    for row in range(v_dim):
        for col in range(rank):
            V[row][col] = 0
    for row in range(w_dim):
        for col in range(rank):
            W[row][col] = 0
    
    # Strassen 7-term decomposition for 2x2x2:
    # M1 = (a00+a11)(b00+b11) -> contributes to c00, c11
    # M2 = (a10+a11)b00       -> contributes to c10, c11
    # M3 = a00(b01-b11)       -> contributes to c01, c11
    # M4 = a11(b10-b00)       -> contributes to c00, c10
    # M5 = (a00+a01)b11       -> contributes to c01, c00
    # M6 = (a10-a00)(b00+b01) -> contributes to c11
    # M7 = (a01-a11)(b10+b11) -> contributes to c00
    
    # For 2x2x4, we have output C with 2 rows (i=0,1) and 4 columns (k=0,1,2,3)
    # W is indexed as W[k*n + i, r] = W[k*2 + i, r]
    # So:
    #   c00 -> W[0*2+0] = W[0]
    #   c01 -> W[1*2+0] = W[2]
    #   c10 -> W[0*2+1] = W[1]
    #   c11 -> W[1*2+1] = W[3]
    #   c02 -> W[2*2+0] = W[4]
    #   c03 -> W[3*2+0] = W[6]
    #   c12 -> W[2*2+1] = W[5]
    #   c13 -> W[3*2+1] = W[7]
    
    # Block 1: columns k=0,1 of B
    # Strassen terms with k indices 0,1
    terms_block1 = [
        # (u_entries, v_entries, w_entries)
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(1,0,1),(0,0,-1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Block 2: columns k=2,3 of B
    terms_block2 = [
        # M1
        ([(0,0,1),(1,1,1)], [(0,2,1),(1,3,1)], [(2,0,1),(3,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,2,1)], [(2,1,1),(3,1,-1)]),
        # M3
        ([(0,0,1)], [(0,3,1),(1,3,-1)], [(3,0,1),(3,1,1)]),
        # M4
        ([(1,1,1)], [(1,2,1),(0,2,-1)], [(2,0,1),(2,1,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,3,1)], [(3,0,1),(2,0,-1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,2,1),(0,3,1)], [(3,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,2,1),(1,3,1)], [(2,0,1)]),
    ]
    
    # Fill in the matrices
    for r in range(7):
        u_ent, v_ent, w_ent = terms_block1[r]
        col = r
        for i, j, val in u_ent:
            U[i*m+j][col] += val
        for j, k, val in v_ent:
            V[j*p+k][col] += val
        for k, i, val in w_ent:
            W[k*n+i][col] += val
    
    for r in range(7):
        u_ent, v_ent, w_ent = terms_block2[r]
        col = 7 + r
        for i, j, val in u_ent:
            U[i*m+j][col] += val
        for j, k, val in v_ent:
            V[j*p+k][col] += val
        for k, i, val in w_ent:
            W[k*n+i][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
