"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x1x5 block (A col 2, B row 2)
    # A indices: i in {0,1}, j=2. U index: i*3 + 2
    # B indices: j=2, k in {0..4}. V index: 2*5 + k
    # C indices: k in {0..4}, i in {0,1}. W index: k*2 + i
    for i in range(2):
        for k in range(5):
            U[i * 3 + 2][col] = 1
            V[2 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: 2x2x5 block (A cols 0,1; B rows 0,1)
    # A indices: i in {0,1}, j in {0,1}. U index: i*3 + j
    # B indices: j in {0,1}, k in {0..4}. V index: j*5 + k
    # C indices: k in {0..4}, i in {0,1}. W index: k*2 + i
    
    # Helper to add Strassen terms for a pair of columns k1, k2
    def add_strassen_block(k1, k2, start_col):
        # M1 = (A00 + A11)(B0k1 + B1k2)
        # M2 = (A10 + A11)(B0k1)
        # M3 = (A00)(B1k2 - B1k1)
        # M4 = (A11)(B0k2 - B1k1)
        # M5 = (A00 + A01)(B1k2)
        # M6 = (A10 - A00)(B0k1 + B0k2)
        # M7 = (A01 - A11)(B1k2 + B1k1)
        
        # Note: My previous Strassen formula was for B = [[B00, B01], [B10, B11]]
        # Here B is [[B0k1, B0k2], [B1k1, B1k2]]
        # Let's map: B00->B0k1, B01->B0k2, B10->B1k1, B11->B1k2
        
        # M1: (A00+A11)(B0k1+B1k2)
        U[0*3+0][start_col+0] = 1
        U[1*3+1][start_col+0] = 1
        V[0*5+k1][start_col+0] = 1
        V[1*5+k2][start_col+0] = 1
        # W contributions for M1:
        # C0k1 += M1, C0k2 -= M1, C1k1 += M1, C1k2 -= M1
        W[k1*2+0][start_col+0] = 1
        W[k2*2+0][start_col+0] = -1
        W[k1*2+1][start_col+0] = 1
        W[k2*2+1][start_col+0] = -1
        
        # M2: (A10+A11)(B0k1)
        U[1*3+0][start_col+1] = 1
        U[1*3+1][start_col+1] = 1
        V[0*5+k1][start_col+1] = 1
        # W contributions for M2:
        # C1k1 += M2, C1k2 -= M2
        W[k1*2+1][start_col+1] = 1
        W[k2*2+1][start_col+1] = -1
        
        # M3: (A00)(B1k2 - B1k1)
        U[0*3+0][start_col+2] = 1
        V[1*5+k2][start_col+2] = 1
        V[1*5+k1][start_col+2] = -1
        # W contributions for M3:
        # C0k2 += M3, C1k1 -= M3
        W[k2*2+0][start_col+2] = 1
        W[k1*2+1][start_col+2] = -1
        
        # M4: (A11)(B0k2 - B1k1)
        U[1*3+1][start_col+3] = 1
        V[0*5+k2][start_col+3] = 1
        V[1*5+k1][start_col+3] = -1
        # W contributions for M4:
        # C0k1 += M4, C0k2 -= M4, C1k1 -= M4, C1k2 += M4
        W[k1*2+0][start_col+3] = 1
        W[k2*2+0][start_col+3] = -1
        W[k1*2+1][start_col+3] = -1
        W[k2*2+1][start_col+3] = 1
        
        # M5: (A00+A01)(B1k2)
        U[0*3+0][start_col+4] = 1
        U[0*3+1][start_col+4] = 1
        V[1*5+k2][start_col+4] = 1
        # W contributions for M5:
        # C0k1 -= M5, C0k2 += M5
        W[k1*2+0][start_col+4] = -1
        W[k2*2+0][start_col+4] = 1
        
        # M6: (A10-A00)(B0k1+B0k2)
        U[1*3+0][start_col+5] = 1
        U[0*3+0][start_col+5] = -1
        V[0*5+k1][start_col+5] = 1
        V[0*5+k2][start_col+5] = 1
        # W contributions for M6:
        # C1k1 += M6, C1k2 += M6
        W[k1*2+1][start_col+5] = 1
        W[k2*2+1][start_col+5] = 1
        
        # M7: (A01-A11)(B1k2+B1k1)
        U[0*3+1][start_col+6] = 1
        U[1*3+1][start_col+6] = -1
        V[1*5+k2][start_col+6] = 1
        V[1*5+k1][start_col+6] = 1
        # W contributions for M7:
        # C0k1 += M7, C0k2 += M7
        W[k1*2+0][start_col+6] = 1
        W[k2*2+0][start_col+6] = 1

    # Add Strassen for cols 0,1
    add_strassen_block(0, 1, col)
    col += 7
    
    # Add Strassen for cols 2,3
    add_strassen_block(2, 3, col)
    col += 7
    
    # Add Schoolbook for col 4
    # C04 = A00 B04 + A01 B14
    # C14 = A10 B04 + A11 B14
    
    # Term 1: A00 B04 -> C04
    U[0*3+0][col] = 1
    V[0*5+4][col] = 1
    W[4*2+0][col] = 1
    col += 1
    
    # Term 2: A01 B14 -> C04
    U[0*3+1][col] = 1
    V[1*5+4][col] = 1
    W[4*2+0][col] = 1
    col += 1
    
    # Term 3: A10 B04 -> C14
    U[1*3+0][col] = 1
    V[0*5+4][col] = 1
    W[4*2+1][col] = 1
    col += 1
    
    # Term 4: A11 B14 -> C14
    U[1*3+1][col] = 1
    V[1*5+4][col] = 1
    W[4*2+1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
