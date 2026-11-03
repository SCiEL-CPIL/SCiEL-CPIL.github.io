"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Build Strassen rank-7 decomposition for 2x2
    # U2: shape (4, 7), indices i*2+j
    # V2: shape (4, 7), indices j*2+k
    # W2: shape (4, 7), indices k*2+i (trace convention)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = (A11+A22)(B11+B22)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1  # C11
    W2[3][0] = 1  # C22
    
    # r=1: M2 = (A21+A22)B11
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1  # C21
    W2[3][1] = -1 # C22
    
    # r=2: M3 = A11(B21-B22)
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[1][2] = 1  # C12
    W2[3][2] = 1  # C22
    
    # r=3: M4 = A22(B12-B11)
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1  # C11
    W2[2][3] = 1  # C21
    
    # r=4: M5 = (A11+A12)B22
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1 # C11
    W2[1][4] = 1  # C12
    
    # r=5: M6 = (A21-A11)(B11+B12)
    U2[0][5] = -1; U2[2][5] = 1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1  # C22
    
    # r=6: M7 = (A12-A22)(B21+B22)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1  # C11
    
    # Now build 4x4 via Kronecker product
    rank = 7 * 7
    U = [[0]*rank for _ in range(n*m)]  # (16, 49)
    V = [[0]*rank for _ in range(m*p)]  # (16, 49)
    W = [[0]*rank for _ in range(p*n)]  # (16, 49)
    
    for a in range(7):
        for b in range(7):
            t = a * 7 + b
            for iA in range(2):
                for jA in range(2):
                    for iB in range(2):
                        for jB in range(2):
                            i = 2*iA + iB
                            j = 2*jA + jB
                            a_idx = i*4 + j
                            aA_idx = iA*2 + jA
                            aB_idx = iB*2 + jB
                            val = U2[aA_idx][a] * U2[aB_idx][b]
                            if val != 0:
                                U[a_idx][t] = val
            
            for jA in range(2):
                for kA in range(2):
                    for jB in range(2):
                        for kB in range(2):
                            j = 2*jA + jB
                            k = 2*kA + kB
                            b_idx = j*4 + k
                            bA_idx = jA*2 + kA
                            bB_idx = jB*2 + kB
                            val = V2[bA_idx][a] * V2[bB_idx][b]
                            if val != 0:
                                V[b_idx][t] = val
            
            for kA in range(2):
                for iA in range(2):
                    for kB in range(2):
                        for iB in range(2):
                            k = 2*kA + kB
                            i = 2*iA + iB
                            c_idx = k*4 + i
                            cA_idx = kA*2 + iA
                            cB_idx = kB*2 + iB
                            val = W2[cA_idx][a] * W2[cB_idx][b]
                            if val != 0:
                                W[c_idx][t] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
