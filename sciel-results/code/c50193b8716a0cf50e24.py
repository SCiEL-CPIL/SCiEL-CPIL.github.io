"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Decompose 6 = 2 * 3
    # Use Strassen (rank 7) for 2x2 and schoolbook (rank 27) for 3x3
    # Total rank = 7 * 27 = 189
    
    # Strassen's 2x2x2 tensor decomposition (rank 7)
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0 (M1) = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=1 (M2) = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # r=2 (M3) = A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1
    
    # r=3 (M4) = A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[2][3] = 1; W2[3][3] = 1
    
    # r=4 (M5) = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = 1; W2[1][4] = 1
    
    # r=5 (M6) = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # r=6 (M7) = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = 1
    
    # Schoolbook 3x3x3 tensor decomposition (rank 27)
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    s = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][s] = 1
                V3[j*3+k][s] = 1
                W3[k*3+i][s] = 1
                s += 1
    
    # Construct 6x6x6 factors via Kronecker product
    rank = 7 * 27
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for r in range(7):
        for s in range(27):
            col = r * 27 + s
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(3):
                        for j2 in range(3):
                            i = 2*i1 + i2
                            j = 2*j1 + j2
                            val = U2[i1*2+j1][r] * U3[i2*3+j2][s]
                            if val != 0:
                                U[i*6+j][col] = val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(3):
                        for k2 in range(3):
                            j = 2*j1 + j2
                            k = 2*k1 + k2
                            val = V2[j1*2+k1][r] * V3[j2*3+k2][s]
                            if val != 0:
                                V[j*6+k][col] = val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(3):
                        for i2 in range(3):
                            k = 2*k1 + k2
                            i = 2*i1 + i2
                            val = W2[k1*2+i1][r] * W3[k2*3+i2][s]
                            if val != 0:
                                W[k*6+i][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
