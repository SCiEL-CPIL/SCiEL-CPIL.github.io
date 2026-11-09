import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # 2x2 Strassen coefficients
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    # A11: i=0,j=0. A22: i=1,j=1.
    U2[0*2+0][0] = 1
    U2[1*2+1][0] = 1
    V2[0*2+0][0] = 1
    V2[1*2+1][0] = 1
    W2[0*2+0][0] = 1 # C11
    W2[1*2+1][0] = 1 # C22
    
    # M2 = (A21+A22)B11
    U2[1*2+0][1] = 1
    U2[1*2+1][1] = 1
    V2[0*2+0][1] = 1
    W2[1*2+0][1] = 1 # C21
    W2[1*2+1][1] = -1 # C22
    
    # M3 = A11(B12-B22)
    U2[0*2+0][2] = 1
    V2[0*2+1][2] = 1
    V2[1*2+1][2] = -1
    W2[0*2+1][2] = 1 # C12
    W2[1*2+1][2] = 1 # C22
    
    # M4 = A22(B21-B11)
    U2[1*2+1][3] = 1
    V2[1*2+0][3] = 1
    V2[0*2+0][3] = -1
    W2[0*2+0][3] = 1 # C11
    W2[1*2+0][3] = 1 # C21
    
    # M5 = (A11+A12)B22
    U2[0*2+0][4] = 1
    U2[0*2+1][4] = 1
    V2[1*2+1][4] = 1
    W2[0*2+0][4] = -1 # C11
    W2[0*2+1][4] = 1 # C12
    
    # M6 = (A21-A11)(B11+B12)
    U2[1*2+0][5] = 1
    U2[0*2+0][5] = -1
    V2[0*2+0][5] = 1
    V2[0*2+1][5] = 1
    W2[1*2+1][5] = 1 # C22
    
    # M7 = (A12-A22)(B21+B22)
    U2[0*2+1][6] = 1
    U2[1*2+1][6] = -1
    V2[1*2+0][6] = 1
    V2[1*2+1][6] = 1
    W2[0*2+0][6] = 1 # C11
    W2[0*2+1][6] = 1 # C12
    
    # Construct 4x4 matrices
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for i in range(4):
        for j in range(4):
            i1 = i // 2
            i2 = i % 2
            j1 = j // 2
            j2 = j % 2
            for r in range(49):
                r1 = r // 7
                r2 = r % 7
                U[i*4+j][r] = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                
    for j in range(4):
        for k in range(4):
            j1 = j // 2
            j2 = j % 2
            k1 = k // 2
            k2 = k % 2
            for r in range(49):
                r1 = r // 7
                r2 = r % 7
                V[j*4+k][r] = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                
    for k in range(4):
        for i in range(4):
            k1 = k // 2
            k2 = k % 2
            i1 = i // 2
            i2 = i % 2
            for r in range(49):
                r1 = r // 7
                r2 = r % 7
                W[k*4+i][r] = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                
    return {"U": U, "V": V, "W": W}
