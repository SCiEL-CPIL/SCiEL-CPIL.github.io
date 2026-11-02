"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition via tensor product of rank-7 Strassen decomposition
    rank = 49
    
    # Define rank-7 decomposition for 2x2 matrices
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11)
    U2[0][0] += 1; U2[3][0] += 1
    V2[0][0] += 1; V2[3][0] += 1
    W2[0][0] += 1; W2[3][0] += 1
    
    # M2: (a10+a11)b00
    U2[2][1] += 1; U2[3][1] += 1
    V2[0][1] += 1
    W2[2][1] += 1; W2[3][1] -= 1
    
    # M3: a00(b10-b11)
    U2[0][2] += 1
    V2[2][2] += 1; V2[3][2] -= 1
    W2[1][2] += 1
    
    # M4: a11(b01-b00)
    U2[3][3] += 1
    V2[1][3] += 1; V2[0][3] -= 1
    W2[0][3] -= 1; W2[3][3] += 1
    
    # M5: (a00+a01)b11
    U2[0][4] += 1; U2[1][4] += 1
    V2[3][4] += 1
    W2[0][4] += 1; W2[1][4] += 1
    
    # M6: (a10-a00)(b00+b10)
    U2[2][5] += 1; U2[0][5] -= 1
    V2[0][5] += 1; V2[2][5] += 1
    W2[2][5] += 1
    
    # M7: (a01-a11)(b11+b01)
    U2[1][6] += 1; U2[3][6] -= 1
    V2[3][6] += 1; V2[1][6] += 1
    W2[0][6] -= 1; W2[3][6] += 1

    # Construct 4x4 decomposition
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1 * 2 + j1][c1] * U2[i2 * 2 + j2][c2]
                            if val != 0:
                                U[i * 4 + j][c] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1 * 2 + k1][c1] * V2[j2 * 2 + k2][c2]
                            if val != 0:
                                V[j * 4 + k][c] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1 * 2 + i1][c1] * W2[k2 * 2 + i2][c2]
                            if val != 0:
                                W[k * 4 + i][c] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
