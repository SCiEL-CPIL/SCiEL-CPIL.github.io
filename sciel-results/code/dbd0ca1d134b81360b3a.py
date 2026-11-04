"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Kronecker product of 3x3 and 2x2 schoolbook decompositions
    # Rank = 9 * 4 = 36
    
    n3, m3, p3 = 3, 3, 3
    n2, m2, p2 = 2, 2, 2
    
    # Build 3x3 schoolbook decomposition
    U3 = [[0]*9 for _ in range(9)]
    V3 = [[0]*9 for _ in range(9)]
    W3 = [[0]*9 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*3+i][col] = 1
                col += 1
    
    # Build 2x2 schoolbook decomposition
    U2 = [[0]*4 for _ in range(4)]
    V2 = [[0]*4 for _ in range(4)]
    W2 = [[0]*4 for _ in range(4)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U2[i*2+j][col] = 1
                V2[j*2+k][col] = 1
                W2[k*2+i][col] = 1
                col += 1
    
    rank = 9 * 4  # 36
    
    U6 = [[0]*rank for _ in range(36)]
    V6 = [[0]*rank for _ in range(36)]
    W6 = [[0]*rank for _ in range(36)]
    
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                r3 = i3*9 + j3*3 + k3
                for i2 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            r2 = i2*4 + j2*2 + k2
                            r = r3*4 + r2
                            
                            row_u = (i3*2+i2)*6 + (j3*2+j2)
                            row_v = (j3*2+j2)*6 + (k3*2+k2)
                            row_w = (k3*2+k2)*6 + (i3*2+i2)
                            
                            U6[row_u][r] = U3[i3*3+j3][r3] * U2[i2*2+j2][r2]
                            V6[row_v][r] = V3[j3*3+k3][r3] * V2[j2*2+k2][r2]
                            W6[row_w][r] = W3[k3*3+i3][r3] * W2[k2*2+i2][r2]
    
    return {"U": U6, "V": V6, "W": W6}
# EVOLVE-BLOCK-END
