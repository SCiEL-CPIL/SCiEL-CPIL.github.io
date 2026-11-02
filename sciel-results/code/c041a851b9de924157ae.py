"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of rank 19: Strassen for 2x2 (rank 7) + Schoolbook for 2x3 (rank 12)
    n, m, p = DIMENSIONS
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # A is 2x2: A[i*m+j] -> U[i*m+j, c]
    # B is 2x5: B[j*p+k] -> V[j*p+k, c]
    # C is 2x5: C[k*n+i] -> W[k*n+i, c]
    
    # Part 1: Strassen for columns 3, 4 of B (indices 2, 3 in 0-based? No, p=5. Indices 3,4 are 3,4)
    # B cols: 0, 1, 2 for 2x3 part. 3, 4 for 2x2 part.
    
    # Strassen for A (2x2) * B_right (2x2)
    # B_right[0,0] = B[0,3], B_right[0,1] = B[0,4]
    # B_right[1,0] = B[1,3], B_right[1,1] = B[1,4]
    
    # A[0,0]=A00, A[0,1]=A01, A[1,0]=A10, A[1,1]=A11
    
    # M1 = (A00+A11)(B03+B14)
    U[0][0] = 1; U[3][0] = 1
    V[3][0] = 1; V[4][0] = 1
    W[3][0] = 1; W[4][0] = 1 # C[0,3] and C[1,4]? No.
    # W[k*n+i, c]. C[0,3] is k=3, i=0 -> W[3*2+0]=W[6]. C[1,4] is k=4, i=1 -> W[4*2+1]=W[9].
    # Strassen outputs:
    # C03 = M1 + M4 - M5 + M7
    # C04 = M3 + M5
    # C13 = M2 + M4
    # C14 = M1 - M2 + M3
    
    # Let's map Strassen products to columns 0-6 of U,V,W
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[3][0] = 1; V[4][0] = 1
    W[6][0] = 1; W[9][0] = 1 # C[0,3] and C[1,4]
    
    # M2 = (A10+A11)B03
    U[2][1] = 1; U[3][1] = 1
    V[3][1] = 1
    W[8][1] = 1; W[9][1] = -1 # C[1,3] and C[1,4]
    
    # M3 = A00(B04-B14)
    U[0][2] = 1
    V[4][2] = 1; V[7][2] = -1
    W[7][2] = 1; W[9][2] = 1 # C[0,4] and C[1,4]
    
    # M4 = A11(B13-B03)
    U[3][3] = 1
    V[6][3] = 1; V[3][3] = -1
    W[6][3] = 1; W[8][3] = 1 # C[0,3] and C[1,3]
    
    # M5 = (A00+A01)B14
    U[0][4] = 1; U[1][4] = 1
    V[7][4] = 1
    W[6][4] = -1; W[7][4] = 1 # C[0,3] and C[0,4]
    
    # M6 = (A10-A00)(B03+B04)
    U[2][5] = 1; U[0][5] = -1
    V[3][5] = 1; V[4][5] = 1
    W[8][5] = 1; W[9][5] = -1 # C[1,3] and C[1,4]
    
    # M7 = (A01-A11)(B13+B14)
    U[1][6] = 1; U[3][6] = -1
    V[6][6] = 1; V[7][6] = 1
    W[6][6] = 1 # C[0,3]
    
    # Part 2: Schoolbook for columns 0, 1, 2 of B (indices 0, 1, 2)
    # 12 products
    
    # C00 = A00*B00 + A01*B10
    # P1 = A00*B00
    U[0][7] = 1; V[0][7] = 1; W[0][7] = 1
    # P2 = A01*B10
    U[1][8] = 1; V[5][8] = 1; W[0][8] = 1
    
    # C01 = A00*B01 + A01*B11
    # P3 = A00*B01
    U[0][9] = 1; V[1][9] = 1; W[1][9] = 1
    # P4 = A01*B11
    U[1][10] = 1; V[6][10] = 1; W[1][10] = 1
    
    # C02 = A00*B02 + A01*B12
    # P5 = A00*B02
    U[0][11] = 1; V[2][11] = 1; W[2][11] = 1
    # P6 = A01*B12
    U[1][12] = 1; V[7][12] = 1; W[2][12] = 1
    
    # C10 = A10*B00 + A11*B10
    # P7 = A10*B00
    U[2][13] = 1; V[0][13] = 1; W[2][13] = 1
    # P8 = A11*B10
    U[3][14] = 1; V[5][14] = 1; W[2][14] = 1
    
    # C11 = A10*B01 + A11*B11
    # P9 = A10*B01
    U[2][15] = 1; V[1][15] = 1; W[3][15] = 1
    # P10 = A11*B11
    U[3][16] = 1; V[6][16] = 1; W[3][16] = 1
    
    # C12 = A10*B02 + A11*B12
    # P11 = A10*B02
    U[2][17] = 1; V[2][17] = 1; W[4][17] = 1
    # P12 = A11*B12
    U[3][18] = 1; V[7][18] = 1; W[4][18] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
