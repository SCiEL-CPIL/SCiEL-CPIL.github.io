"""Matrix multiplication 2x4 x 4x8 with rank-56 Strassen-based decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-56 decomposition using block Strassen for 2x2 sub-blocks.
    
    A (2x4) partitioned into A0 (cols 0,1) and A1 (cols 2,3), each 2x2.
    B (4x8) partitioned into 4 blocks Bl (4x2) for l=0..3, each split into
    Bl_top (rows 0,1) and Bl_bottom (rows 2,3), each 2x2.
    
    C[:, 2l:2l+2] = A0 * Bl_top + A1 * Bl_bottom
    Each 2x2*2x2 -> 2x2 uses Strassen rank 7.
    Total: 4 * 2 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen 7 products for 2x2 * 2x2 -> 2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Each product: list of (local_A_idx, coeff) and (local_B_idx, coeff)
    # Local 2x2 indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_A = [
        [(0, 1), (3, 1)],       # M1
        [(2, 1), (3, 1)],       # M2
        [(0, 1)],               # M3
        [(3, 1)],               # M4
        [(0, 1), (1, 1)],       # M5
        [(2, 1), (0, -1)],      # M6
        [(1, 1), (3, -1)],      # M7
    ]
    strassen_B = [
        [(0, 1), (3, 1)],       # M1
        [(0, 1)],               # M2
        [(2, 1), (3, -1)],      # M3
        [(1, 1), (0, -1)],      # M4
        [(3, 1)],               # M5
        [(0, 1), (2, 1)],       # M6
        [(1, 1), (3, 1)],       # M7
    ]
    
    # For each output position, which products contribute and with what coefficient
    # output_combos[c_local] = list of (product_idx, coeff)
    # c_local: 0=c00, 1=c01, 2=c10, 3=c11
    output_combos = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # c00
        [(2, 1), (4, 1)],                     # c01
        [(1, 1), (3, 1)],                     # c10
        [(0, 1), (1, -1), (2, 1), (5, 1)],   # c11
    ]
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    for l in range(4):  # 4 column blocks of B (each 2 cols: 2l, 2l+1)
        for term in range(2):  # term 0: A0*Bl_top, term 1: A1*Bl_bottom
            if term == 0:
                a_col_offset = 0   # A cols 0,1
                b_row_offset = 0   # B rows 0,1
            else:
                a_col_offset = 2   # A cols 2,3
                b_row_offset = 2   # B rows 2,3
            
            for s in range(7):  # 7 Strassen products
                # Fill U
                for (a_local, a_coeff) in strassen_A[s]:
                    a_row = a_local // 2
                    a_col = a_col_offset + (a_local % 2)
                    a_idx = a_row * m + a_col
                    U[a_idx][col] = a_coeff
                
                # Fill V
                for (b_local, b_coeff) in strassen_B[s]:
                    b_row = b_row_offset + (b_local // 2)
                    b_col = 2 * l + (b_local % 2)
                    b_idx = b_row * p + b_col
                    V[b_idx][col] = b_coeff
                
                # Fill W: accumulate contributions from this product to output positions
                for c_local in range(4):
                    for (prod_idx, coeff) in output_combos[c_local]:
                        if prod_idx == s:
                            # Determine global C position
                            c_row = c_local // 2
                            c_col = 2 * l + (c_local % 2)
                            w_idx = c_col * n + c_row
                            W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
