"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 1. Strassen for 2x2x2 sub-block (i,j,k in {0,1})
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for U (A part) and V (B part)
    # U index: i*m + j
    # V index: j*p + k
    
    # r=0: M1
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*3+0][0] = 1; V[1*3+1][0] = 1
    W[0*2+0][0] = 1 # C00 += M1
    W[1*2+1][0] = 1 # C11 += M1
    
    # r=1: M2
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*3+0][1] = 1
    W[1*2+0][1] = 1 # C10 += M2
    W[1*2+1][1] = -1 # C11 -= M2
    
    # r=2: M3
    U[0*3+0][2] = 1
    V[0*3+1][2] = 1; V[1*3+1][2] = -1
    W[0*2+1][2] = 1 # C01 += M3
    W[1*2+1][2] = 1 # C11 += M3
    
    # r=3: M4
    U[1*3+1][3] = 1
    V[1*3+0][3] = 1; V[0*3+0][3] = -1
    W[0*2+0][3] = 1 # C00 += M4
    W[1*2+0][3] = 1 # C10 += M4
    
    # r=4: M5
    U[0*3+0][4] = 1; U[0*3+1][4] = 1
    V[1*3+1][4] = 1
    W[0*2+0][4] = -1 # C00 -= M5
    W[0*2+1][4] = 1 # C01 += M5
    
    # r=5: M6
    U[1*3+0][5] = 1; U[0*3+0][5] = -1
    V[0*3+0][5] = 1; V[0*3+1][5] = 1
    W[1*2+1][5] = 1 # C11 += M6
    
    # r=6: M7
    U[0*3+1][6] = 1; U[1*3+1][6] = -1
    V[1*3+0][6] = 1; V[1*3+1][6] = 1
    W[0*2+0][6] = 1 # C00 += M7
    
    # 2. Schoolbook for j=2 (A_i2 * B_2k)
    # i in {0,1}, k in {0,1,2}
    col = 7
    for i in range(2):
        for k in range(3):
            U[i*3+2][col] = 1
            V[2*3+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    # 3. Schoolbook for k=2, j in {0,1} (A_ij * B_j2)
    # i in {0,1}, j in {0,1}
    for i in range(2):
        for j in range(2):
            U[i*3+j][col] = 1
            V[j*3+2][col] = 1
            W[2*2+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
