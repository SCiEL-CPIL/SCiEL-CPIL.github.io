"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Block-Strassen decomposition for 6x6 matrix multiplication.
    # Partition into 3x3 blocks of 2x2 matrices. Each 2x2 product uses Strassen (rank 7).
    # Total rank = 9 * 7 = 63... wait, no. For each (i,j,k) block position, we need 7 factors.
    # 3*3*3 = 27 block positions, each with 7 factors = 189.
    
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Standard Strassen factors for 2x2: C = A*B
    # A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]], C = [[C00, C01],[C10, C11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each factor defined as (U_entries, V_entries, W_entries)
    # entries are lists of ((local_row, local_col), coeff)
    factors = [
        # M1
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        # M2
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1)]),
        # M3
        ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1)]),
        # M4
        ([((1,1),1)], [((1,0),1),((0,0),-1)], [((1,0),1)]),
        # M5
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1)]),
        # M6
        ([((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)], [((1,1),1)]),
        # M7
        ([((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)], [((0,1),1)]),
    ]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for u_entries, v_entries, w_entries in factors:
                    for (r, c), coeff in u_entries:
                        U[(i*2+r)*6 + (j*2+c)][col] += coeff
                    for (r, c), coeff in v_entries:
                        V[(j*2+r)*6 + (k*2+c)][col] += coeff
                    for (r, c), coeff in w_entries:
                        W[(k*2+r)*6 + (i*2+c)][col] += coeff
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
