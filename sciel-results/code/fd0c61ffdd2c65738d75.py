"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 decomposition of matmul(2,4,6) using Strassen on 2x2 sub-blocks.
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen 2x2x2:
    strassen_U = [
        [1, 0, 0, 1],   # M1: X00+X11
        [0, 0, 1, 1],   # M2: X10+X11
        [1, 0, 0, 0],   # M3: X00
        [0, 0, 0, 1],   # M4: X11
        [1, 1, 0, 0],   # M5: X00+X01
        [-1, 0, 1, 0],  # M6: X10-X00
        [0, 1, 0, -1],  # M7: X01-X11
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: Y00+Y11
        [1, 0, 0, 0],   # M2: Y00
        [0, 1, 0, -1],  # M3: Y01-Y11
        [0, 0, 1, -1],  # M4: Y10-Y00
        [0, 0, 0, 1],   # M5: Y11
        [1, 1, 0, 0],   # M6: Y00+Y01
        [0, 0, 1, 1],   # M7: Y10+Y11
    ]
    strassen_W_vals = [
        [1, 0, 0, 1],   # Z00: M1+M4-M5+M7
        [0, 0, 1, 1],   # Z01: M3+M5
        [0, 1, 1, 0],   # Z10: M2+M4
        [1, -1, 1, 1],  # Z11: M1-M2+M3+M6
    ]
    w_row_offsets = [0, 2, 1, 3]  # W index offsets for Z00, Z01, Z10, Z11
    
    def add_strassen_block(u_indices, v_indices, w_base, col_offset, sign=1):
        """
        u_indices: list of 4 U row indices for X00, X01, X10, X11
        v_indices: list of 4 V row indices for Y00, Y01, Y10, Y11
        w_base: base W index (for the 2x2 result block)
        """
        for s in range(7):
            for r in range(4):
                if strassen_U[s][r] != 0:
                    U[u_indices[r]][col_offset + s] += sign * strassen_U[s][r]
                if strassen_V[s][r] != 0:
                    V[v_indices[r]][col_offset + s] += sign * strassen_V[s][r]
                for r in range(4):
                    if strassen_W_vals[r][s] != 0:
                        W[w_base + w_row_offsets[r]][col_offset + s] += sign * strassen_W_vals[r][s]
    
    # A (2x4): U index = i*4+j
    # A1 (cols 0,1): X00=A[0][0]->U[0], X01=A[0][1]->U[1], X10=A[1][0]->U[4], X11=A[1][1]->U[5]
    # A2 (cols 2,3): X00=A[0][2]->U[2], X01=A[0][3]->U[3], X10=A[1][2]->U[6], X11=A[1][3]->U[7]
    
    # B (4x6): V index = j*6+k
    # B1 (rows 0,1): 
    #   B1a (cols 0,1): Y00=B[0][0]->V[0], Y01=B[0][1]->V[1], Y10=B[1][0]->V[6], Y11=B[1][1]->V[7]
    #   B1b (cols 2,3): Y00=B[0][2]->V[2], Y01=B[0][3]->V[3], Y10=B[1][2]->V[8], Y11=B[1][3]->V[9]
    #   B1c (cols 4,5): Y00=B[0][4]->V[4], Y01=B[0][5]->V[5], Y10=B[1][4]->V[10], Y11=B[1][5]->V[11]
    # B2 (rows 2,3):
    #   B2a (cols 0,1): Y00=B[2][0]->V[12], Y01=B[2][1]->V[13], Y10=B[3][0]->V[18], Y11=B[3][1]->V[19]
    #   B2b (cols 2,3): Y00=B[2][2]->V[14], Y01=B[2][3]->V[15], Y10=B[3][2]->V[20], Y11=B[3][3]->V[21]
    #   B2c (cols 4,5): Y00=B[2][4]->V[16], Y01=B[2][5]->V[17], Y10=B[3][4]->V[22], Y11=B[3][5]->V[23]
    
    # C (2x6): W index = k*2+i
    # C1 = A1*B1, C2 = A2*B2
    # C1[:, 0:2]: W indices 0,1,2,3 (k=0,1; i=0,1)
    # C1[:, 2:4]: W indices 4,5,6,7
    # C1[:, 4:6]: W indices 8,9,10,11
    
    # A1*B1a
    add_strassen_block([0,1,4,5], [0,1,6,7], 0, 0)
    # A1*B1b
    add_strassen_block([0,1,4,5], [2,3,8,9], 4, 7)
    # A1*B1c
    add_strassen_block([0,1,4,5], [4,5,10,11], 8, 14)
    
    # A2*B2a
    add_strassen_block([2,3,6,7], [12,13,18,19], 0, 21)
    # A2*B2b
    add_strassen_block([2,3,6,7], [14,15,20,21], 4, 28)
    # A2*B2c
    add_strassen_block([2,3,6,7], [16,17,22,23], 8, 35)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
