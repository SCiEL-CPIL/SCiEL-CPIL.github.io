"""
Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
Uses the Kronecker product of the rank-7 Strassen decomposition for 2x2 matrices.
Rank = 7 * 7 = 49.
"""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin, construction is very fast
    
    n, m, p = DIMENSIONS
    # Construct 2x2 Strassen decomposition (Rank 7)
    # U2, V2, W2 are 4x7 matrices
    # Indices: U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # c=0: M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # c=1: M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # c=2: M3 = A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # c=3: M4 = A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[1][3] = 1; W2[0][3] = 1
    
    # c=4: M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # c=5: M6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # c=6: M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Construct 4x4 decomposition (Rank 49)
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1*2 + j1][c1]
                    if u1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2*2 + j2][c2]
                            if u2 == 0: continue
                            val = u1 * u2
                            i = i1*2 + i2
                            j = j1*2 + j2
                            U[i*4 + j][c] = val
                            
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V2[j1*2 + k1][c1]
                    if v1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v2 = V2[j2*2 + k2][c2]
                            if v2 == 0: continue
                            val = v1 * v2
                            j = j1*2 + j2
                            k = k1*2 + k2
                            V[j*4 + k][c] = val
                            
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W2[k1*2 + i1][c1]
                    if w1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w2 = W2[k2*2 + i2][c2]
                            if w2 == 0: continue
                            val = w1 * w2
                            k = k1*2 + k2
                            i = i1*2 + i2
                            W[k*4 + i][c] = val
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
