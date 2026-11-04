"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Use tensor product of Strassen (2x2, rank 7) and Schoolbook (3x3, rank 27)
    # Total rank = 7 * 27 = 189
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    # i,j,k in {0,1}, r in {0..6}
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: m0
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=1: m1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[0][1] = 1 # W[k*2+i], c10 -> i=1, k=0 -> idx 0*2+1=1? No.
    # W index is k*n+i. For 2x2, n=2. W[k*2+i].
    # c10 is i=1, k=0. Index = 0*2+1 = 1.
    # So W2[1][1] = 1.
    W2[1][1] = 1
    
    # r=2: m2
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    # c01 is i=0, k=1. Index = 1*2+0 = 2.
    W2[2][2] = 1
    
    # r=3: m3
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    # c00 is i=0, k=0. Index = 0*2+0 = 0.
    W2[0][3] = 1
    
    # r=4: m4
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    # c11 is i=1, k=1. Index = 1*2+1 = 3.
    # c10 is i=1, k=0. Index = 0*2+1 = 1.
    W2[3][4] = 1
    W2[1][4] = 1
    
    # r=5: m5
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    # c00 is i=0, k=0. Index 0. Coeff -1.
    W2[0][5] = -1
    # c01 is i=0, k=1. Index 2. Coeff 1.
    W2[2][5] = 1
    
    # r=6: m6
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    # c00 is i=0, k=0. Index 0. Coeff -1.
    W2[0][6] = -1
    
    # Schoolbook factors for 3x3 (Rank 27)
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*3+i][col] = 1
                col += 1
                
    # Combine
    rank = 7 * 27
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for r2 in range(7):
        for r3 in range(27):
            r = r2 * 27 + r3
            for i2 in range(2):
                for j2 in range(2):
                    for i3 in range(3):
                        for j3 in range(3):
                            i = i2 * 3 + i3
                            j = j2 * 3 + j3
                            val = U2[i2*2+j2][r2] * U3[i3*3+j3][r3]
                            U[i*6+j][r] = val
                            
                            val_v = V2[j2*2+0][r2] # Placeholder, need k3
                            # V index is j*6+k. j = j2*3+j3, k = k2*3+k3
                            # V2 index is j2*2+k2. V3 index is j3*3+k3.
                            
    # Correct Combination Loop
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for r2 in range(7):
        for r3 in range(27):
            r = r2 * 27 + r3
            # U
            for i2 in range(2):
                for j2 in range(2):
                    u2_val = U2[i2*2+j2][r2]
                    if u2_val == 0: continue
                    for i3 in range(3):
                        for j3 in range(3):
                            u3_val = U3[i3*3+j3][r3]
                            if u3_val == 0: continue
                            i = i2*3 + i3
                            j = j2*3 + j3
                            U[i*6+j][r] = u2_val * u3_val
                            
            # V
            for j2 in range(2):
                for k2 in range(2):
                    v2_val = V2[j2*2+k2][r2]
                    if v2_val == 0: continue
                    for j3 in range(3):
                        for k3 in range(3):
                            v3_val = V3[j3*3+k3][r3]
                            if v3_val == 0: continue
                            j = j2*3 + j3
                            k = k2*3 + k3
                            V[j*6+k][r] = v2_val * v3_val
                            
            # W
            for k2 in range(2):
                for i2 in range(2):
                    w2_val = W2[k2*2+i2][r2]
                    if w2_val == 0: continue
                    for k3 in range(3):
                        for i3 in range(3):
                            w3_val = W3[k3*3+i3][r3]
                            if w3_val == 0: continue
                            k = k2*3 + k3
                            i = i2*3 + i3
                            W[k*6+i][r] = w2_val * w3_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
