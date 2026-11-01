"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 11: 7 (Strassen for 2x2x2) + 4 (Schoolbook for 2x2x1)
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (2x2): 0:00, 1:01, 2:10, 3:11
    # Indices for B (2x3): 0:00, 1:01, 2:02, 3:10, 4:11, 5:12
    # Indices for C (2x3, k*n+i): 0:00, 1:10, 2:01, 3:11, 4:02, 5:12
    
    # Strassen for columns 0, 1 of B
    # P0 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1 # C00
    W[3][0] = 1 # C11
    
    # P1 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1 # C10
    W[3][1] = -1 # C11
    
    # P2 = A00(B10-B11)
    U[0][2] = 1
    V[3][2] = 1; V[4][2] = -1
    W[2][2] = 1 # C01
    W[3][2] = 1 # C11
    
    # P3 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1 # C00
    W[3][3] = 1 # C11
    
    # P4 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1 # C00
    W[2][4] = 1 # C01
    
    # P5 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = 1 # C10
    W[3][5] = 1 # C11
    
    # P6 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1 # C00
    W[2][6] = 0 # C01 (P6 does not contribute to C01 in standard Strassen? 
                  # C01 = P2+P4. C11 = P0-P1+P2+P5. C00 = P0+P3-P4+P6. C10 = P1+P5.
                  # So P6 contributes only to C00.)
    
    # Schoolbook for column 2 of B (indices 2, 5 in V)
    # C02 = A00*B02 + A01*B12 (W index 4)
    # C12 = A10*B02 + A11*B12 (W index 5)
    
    # Q0 = A00 * B02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # Q1 = A01 * B12
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # Q2 = A10 * B02
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # Q3 = A11 * B12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
