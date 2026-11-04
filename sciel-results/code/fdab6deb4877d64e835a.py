import numpy as np

def construct():
    # 2x2x2 Strassen decomposition
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 0: m1 = (a+d)(e+h)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 1: m2 = (c+d)e
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # Term 2: m3 = a(f-h)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1
    
    # Term 3: m4 = d(g-c)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[2][3] = 1
    
    # Term 4: m5 = (a+b)h
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Term 5: m6 = (c-d)(f+h)
    U2[2][5] = 1; U2[3][5] = -1
    V2[1][5] = 1; V2[3][5] = 1
    W2[3][5] = 1
    
    # Term 6: m7 = (a-c)(e+g)
    U2[0][6] = 1; U2[2][6] = -1
    V2[0][6] = 1; V2[2][6] = 1
    W2[0][6] = 1; W2[2][6] = -1
    
    # Build 4x4x4 via tensor product
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for r in range(7):
        for s in range(7):
            c = r*7 + s
            # U
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            I = 2*i1 + i0
                            J = 2*j1 + j0
                            val = U2[i1*2+j1][r] * U2[i0*2+j0][s]
                            if val != 0:
                                U[I*4+J][c] += val
            # V
            for j1 in range(2):
                for j0 in range(2):
                    for k1 in range(2):
                        for k0 in range(2):
                            J = 2*j1 + j0
                            K = 2*k1 + k0
                            val = V2[j1*2+k1][r] * V2[j0*2+k0][s]
                            if val != 0:
                                V[J*4+K][c] += val
            # W
            for k1 in range(2):
                for k0 in range(2):
                    for i1 in range(2):
                        for i0 in range(2):
                            K = 2*k1 + k0
                            I = 2*i1 + i0
                            val = W2[k1*2+i1][r] * W2[k0*2+i0][s]
                            if val != 0:
                                W[K*4+I][c] += val
    
    return {"U": U, "V": V, "W": W}
