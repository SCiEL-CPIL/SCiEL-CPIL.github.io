"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C11,C12],[C21,C22]]
    # P1=(a+d)(e+h), P2=(c+d)e, P3=a(f-h), P4=d(g-e), P5=(a+b)h, P6=(c-a)(e+f), P7=(b-d)(g+h)
    # C11=P1+P4-P5+P7, C12=P3+P5, C21=P2+P4, C22=P1-P2+P3+P6
    
    U_str = [
        [1, 0, 1, 0, 1, -1, 0],   # a
        [0, 0, 0, 0, 1, 0, 1],    # b
        [0, 1, 0, 0, 0, 1, 0],    # c
        [1, 1, 0, 1, 0, 0, -1],   # d
    ]
    V_str = [
        [1, 1, 0, -1, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 1, 0],    # f
        [0, 0, 0, 1, 0, 0, 1],    # g
        [1, 0, -1, 0, 1, 0, 1],   # h
    ]
    W_str = [
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0],   # C22
    ]
    
    # Build decomposition for (2x2)*(2x8): A is 2x2, B is 2x8, C is 2x8
    # Split B into 4 blocks of 2x2: B = [B1|B2|B3|B4]
    # C = [A*B1 | A*B2 | A*B3 | A*B4]
    # Shared Strassen for A: 7 combos. For each Bi: 7 combos. Total: 28 products.
    
    def build_2x2_2x8():
        """Returns U (4x28), V (16x28), W (16x28) for (2x2)*(2x8)."""
        # U: 4 rows (a,b,c,d), 28 cols (7 per block, same pattern repeated)
        U = []
        for row in U_str:
            U.append(row * 4)  # repeat each of the 7 columns 4 times
        
        # V: 16 rows (B has 2x8=16 entries), 28 cols
        # B layout: row-major 2x8: [e0,e1,e2,e3,e4,e5,e6,e7, e8,e9,e10,e11,e12,e13,e14,e15]
        # B1 = [[e0,e1],[e8,e9]], B2 = [[e2,e3],[e10,e11]], B3 = [[e4,e5],[e12,e13]], B4 = [[e6,e7],[e14,e15]]
        # For each block t (0..3), columns 7*t..7*t+6 use V_str pattern on the 4 entries of Bt
        V = [[0]*28 for _ in range(16)]
        # Block t uses B entries: row0: [2*t, 2*t+1], row1: [8+2*t, 8+2*t+1]
        for t in range(4):
            col_offset = 7 * t
            # V_str rows correspond to e,f,g,h of a 2x2 block
            # e -> B[0, 2*t], f -> B[0, 2*t+1], g -> B[1, 2*t], h -> B[1, 2*t+1]
            # In flat index: B[0, 2*t] = 2*t, B[0, 2*t+1] = 2*t+1, B[1, 2*t] = 8+2*t, B[1, 2*t+1] = 8+2*t+1
            for r in range(4):
                for c in range(7):
                    val = V_str[r][c]
                    if val != 0:
                        flat_idx = [2*t, 2*t+1, 8+2*t, 8+2*t+1][r]
                        V[flat_idx][col_offset + c] = val
        
        # W: 16 rows (C has 2x8=16 entries), 28 cols
        # C layout: row-major 2x8: [C00,C01,C02,C03,C04,C05,C06,C07, C10,C11,C12,C13,C14,C15,C16,C17]
        # C = [C1 | C2 | C3 | C4] where Ct is 2x2
        # For block t, columns 7*t..7*t+6 use W_str pattern on the 4 entries of Ct
        W = [[0]*28 for _ in range(16)]
        for t in range(4):
            col_offset = 7 * t
            # W_str rows correspond to C11,C12,C21,C22 of a 2x2 block
            # C11 -> C[0, 2*t], C12 -> C[0, 2*t+1], C21 -> C[1, 2*t], C22 -> C[1, 2*t+1]
            for r in range(4):
                for c in range(7):
                    val = W_str[r][c]
                    if val != 0:
                        flat_idx = [2*t, 2*t+1, 8+2*t, 8+2*t+1][r]
                        W[flat_idx][col_offset + c] = val
        
        return U, V, W
    
    U_2x2_2x8, V_2x2_2x8, W_2x2_2x8 = build_2x2_2x8()
    
    # Now build the full 2x4 * 4x8 decomposition
    # A (2x4) = [A1 | A2] where A1 = A[:,0:2] (2x2), A2 = A[:,2:4] (2x2)
    # B (4x8) = [B1; B2] where B1 = B[0:2,:] (2x8), B2 = B[2:4,:] (2x8)
    # C = A1*B1 + A2*B2
    # 
    # A flat index: row-major 2x4: [A00,A01,A02,A03, A10,A11,A12,A13]
    # A1 = [[A00,A01],[A10,A11]], A2 = [[A02,A03],[A12,A13]]
    # 
    # B flat index: row-major 4x8: [B00..B07, B10..B17, B20..B27, B30..B37]
    # B1 = B[0:2,:] = [[B00..B07],[B10..B17]] (2x8)
    # B2 = B[2:4,:] = [[B20..B27],[B30..B37]] (2x8)
    # 
    # C flat index: row-major 2x8: [C00..C07, C10..C17]
    # C = A1*B1 + A2*B2, both are 2x8
    
    rank = 56
    U = [[0]*rank for _ in range(8)]   # n*m = 2*4 = 8
    V = [[0]*rank for _ in range(32)]  # m*p = 4*8 = 32
    W = [[0]*rank for _ in range(16)]  # p*n = 8*2 = 16
    
    # Term 1: A1*B1
    # A1 flat indices in A: A00=0, A01=1, A10=4, A11=5
    # B1 flat indices in B: B1[0,0..7] = B[0,0..7] = 0..7, B1[1,0..7] = B[1,0..7] = 8..15
    # C1 = A1*B1, C1[0,0..7] = C[0,0..7] = 0..7, C1[1,0..7] = C[1,0..7] = 8..15
    
    for c in range(28):
        # U: map U_2x2_2x8 rows (a,b,c,d) to A1 flat indices
        U_flat = [0, 1, 4, 5]  # a=A00, b=A01, c=A10, d=A11
        for r in range(4):
            U[U_flat[r]][c] += U_2x2_2x8[r][c]
        
        # V: map V_2x2_2x8 rows to B1 flat indices
        # V_2x2_2x8 has 16 rows for B (2x8). B1 in B has flat indices 0..15
        for r in range(16):
            V[r][c] += V_2x2_2x8[r][c]
        
        # W: map W_2x2_2x8 rows to C flat indices
        # W_2x2_2x8 has 16 rows for C (2x8). C1 in C has flat indices 0..15
        for r in range(16):
            W[r][c] += W_2x2_2x8[r][c]
    
    # Term 2: A2*B2
    # A2 flat indices in A: A02=2, A03=3, A12=6, A13=7
    # B2 flat indices in B: B2[0,0..7] = B[2,0..7] = 16..23, B2[1,0..7] = B[3,0..7] = 24..31
    # C2 = A2*B2, C2[0,0..7] = C[0,0..7] = 0..7, C2[1,0..7] = C[1,0..7] = 8..15
    
    offset = 28
    for c in range(28):
        # U: map U_2x2_2x8 rows to A2 flat indices
        U_flat = [2, 3, 6, 7]  # a=A02, b=A03, c=A12, d=A13
        for r in range(4):
            U[U_flat[r]][offset + c] += U_2x2_2x8[r][c]
        
        # V: map V_2x2_2x8 rows to B2 flat indices (16..31)
        for r in range(16):
            V[16 + r][offset + c] += V_2x2_2x8[r][c]
        
        # W: map W_2x2_2x8 rows to C flat indices (0..15)
        for r in range(16):
            W[r][offset + c] += W_2x2_2x8[r][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
