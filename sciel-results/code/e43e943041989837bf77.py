import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 189
    rank = 189
    
    # 2x2 Strassen decomposition
    # U2, V2, W2 are 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = 1
    
    # r=3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # r=4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # r=5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # r=6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # r=7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # 3x3 Schoolbook decomposition
    # U3, V3, W3 are 9x27
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    s = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3 + j][s] = 1
                V3[j*3 + k][s] = 1
                W3[k*3 + i][s] = 1
                s += 1
                
    # Construct U, V, W of size 6x189 (actually 36x189 for U/V, 36x189 for W? No)
    # U shape: (n*m, rank) = (36, 189)
    # V shape: (m*p, rank) = (36, 189)
    # W shape: (p*n, rank) = (36, 189)
    
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for r in range(7):
        for s in range(27):
            col = r * 27 + s
            
            # U
            # Row index I = i * 6 + j
            # i = 3*ib + il
            # j = 3*jb + jl
            # U2 index: ib * 2 + jb
            # U3 index: il * 3 + jl
            for ib in range(2):
                for jb in range(2):
                    u2_val = U2[ib*2 + jb][r]
                    if u2_val == 0: continue
                    for il in range(3):
                        for jl in range(3):
                            u3_val = U3[il*3 + jl][s]
                            if u3_val == 0: continue
                            i = 3*ib + il
                            j = 3*jb + jl
                            I = i * 6 + j
                            U[I][col] += u2_val * u3_val
                            
            # V
            # Row index J = j * 6 + k
            # j = 3*jb + jl
            # k = 3*kb + kl
            # V2 index: jb * 2 + kb
            # V3 index: jl * 3 + kl
            for jb in range(2):
                for kb in range(2):
                    v2_val = V2[jb*2 + kb][r]
                    if v2_val == 0: continue
                    for jl in range(3):
                        for kl in range(3):
                            v3_val = V3[jl*3 + kl][s]
                            if v3_val == 0: continue
                            j = 3*jb + jl
                            k = 3*kb + kl
                            J = j * 6 + k
                            V[J][col] += v2_val * v3_val
                            
            # W
            # Row index K = k * 6 + i
            # k = 3*kb + kl
            # i = 3*ib + il
            # W2 index: kb * 2 + ib
            # W3 index: kl * 3 + il
            for kb in range(2):
                for ib in range(2):
                    w2_val = W2[kb*2 + ib][r]
                    if w2_val == 0: continue
                    for kl in range(3):
                        for il in range(3):
                            w3_val = W3[kl*3 + il][s]
                            if w3_val == 0: continue
                            k = 3*kb + kl
                            i = 3*ib + il
                            K = k * 6 + i
                            W[K][col] += w2_val * w3_val
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
