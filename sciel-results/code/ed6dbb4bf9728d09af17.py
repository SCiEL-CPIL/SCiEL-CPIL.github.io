import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Sub-problem 1: 2x2 by 2x3 (Rank 6 Schoolbook)
    # U1: (2*2, 6)
    # V1: (2*3, 6)
    # W1: (3*2, 6)
    r1 = 6
    U1 = [[0]*r1 for _ in range(4)]
    V1 = [[0]*r1 for _ in range(6)]
    W1 = [[0]*r1 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U1[i*2+j][col] = 1
                V1[j*3+k][col] = 1
                W1[k*2+i][col] = 1
                col += 1
                
    # Sub-problem 2: 2x2 by 2x2 (Rank 7 Laderman)
    # U2: (2*2, 7)
    # V2: (2*2, 7)
    # W2: (2*2, 7)
    r2 = 7
    U2 = [[0]*r2 for _ in range(4)]
    V2 = [[0]*r2 for _ in range(4)]
    W2 = [[0]*r2 for _ in range(4)]
    
    # Laderman coefficients
    # M1: a11 b11
    U2[0][0] = 1; V2[0][0] = 1; W2[0][0] = 1
    # M2: a22 (b11+b12)
    U2[3][1] = 1; V2[0][1] = 1; V2[1][1] = 1; W2[1][1] = 1; W2[3][1] = 1 # c21, c22?
    # Let's use the explicit W2 derived above.
    # W2 rows: (0,0), (0,1), (1,0), (1,1)
    # c11 (0,0): M1 + M4 - M7 -> 1, 0, 0, 1, 0, 0, -1
    # c12 (0,1): M3 + M7 -> 0, 0, 1, 0, 0, 0, 1
    # c21 (1,0): M6 + M4 -> 0, 0, 0, 1, 0, 1, 0
    # c22 (1,1): M2 - M4 - M7 -> 0, 1, 0, -1, 0, 0, -1
    
    W2[0][0] = 1; W2[0][3] = 1; W2[0][6] = -1
    W2[1][2] = 1; W2[1][6] = 1
    W2[2][3] = 1; W2[2][5] = 1
    W2[3][1] = 1; W2[3][3] = -1; W2[3][6] = -1
    
    # U2 and V2 must match these W2 coefficients.
    # M1 (col 0): a11, b11
    U2[0][0] = 1; V2[0][0] = 1
    # M2 (col 1): a22, b11+b12
    U2[3][1] = 1; V2[0][1] = 1; V2[1][1] = 1
    # M3 (col 2): a11, b21+b22
    U2[0][2] = 1; V2[2][2] = 1; V2[3][2] = 1
    # M4 (col 3): a22, b21-b11
    U2[3][3] = 1; V2[2][3] = 1; V2[0][3] = -1
    # M5 (col 4): a12, b21+b22
    U2[1][4] = 1; V2[2][4] = 1; V2[3][4] = 1
    # M6 (col 5): a21, b11+b12
    U2[2][5] = 1; V2[0][5] = 1; V2[1][5] = 1
    # M7 (col 6): a11+a22, b22-b11
    U2[0][6] = 1; U2[3][6] = 1; V2[3][6] = 1; V2[0][6] = -1

    rank = r1 * r2
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            U[i*m+j][c] = U1[i1*2+j1][c1] * U2[i2*2+j2][c2]
                            
                            for k1 in range(3):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    V[j*p+k][c] = V1[j1*3+k1][c1] * V2[j2*2+k2][c2]
                                    W[k*n+i][c] = W1[k1*2+i1][c1] * W2[k2*2+i2][c2]
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
