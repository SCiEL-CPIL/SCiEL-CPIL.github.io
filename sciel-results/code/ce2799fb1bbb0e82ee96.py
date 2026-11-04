"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Kronecker product of Strassen (2x2x2, rank 7) and schoolbook (2x2x3, rank 12)
    # giving 4x4x6 with rank 7*12 = 84
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2 (rank 7)
    # U_str[4,7], V_str[4,7], W_str[4,7]
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_str = [
        [1, 0, 1, 0, 1, -1, 0],   # a (i=0,j=0)
        [0, 0, 0, 0, 1, 0, 1],    # b (i=0,j=1)
        [0, 1, 0, 0, 0, 1, 0],    # c (i=1,j=0)
        [1, 1, 0, 1, 0, 0, -1]    # d (i=1,j=1)
    ]
    V_str = [
        [1, 1, 0, 0, 0, 1, 0],   # e (j=0,k=0)
        [0, 0, 1, 0, 0, 1, 0],   # f (j=0,k=1)
        [0, 0, 0, 1, 0, 0, 1],   # g (j=1,k=0)
        [1, 0, -1, -1, 1, 0, 1]  # h (j=1,k=1)
    ]
    W_str = [
        [1, 0, 0, 1, -1, 0, 1],   # C11 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],    # C12 (k=0,i=1)
        [0, 1, 0, 1, 0, 0, 0],    # C21 (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 1]    # C22 (k=1,i=1)
    ]
    
    # Schoolbook for 2x2x3 (rank 12)
    # U_sb[4,12], V_sb[6,12], W_sb[6,12]
    # n=2, m=2, p=3
    U_sb = [[0]*12 for _ in range(4)]
    V_sb = [[0]*12 for _ in range(6)]
    W_sb = [[0]*12 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U_sb[i*2+j][col] = 1
                V_sb[j*3+k][col] = 1
                W_sb[k*2+i][col] = 1
                col += 1
    
    # Kronecker product: rank = 7*12 = 84
    # U_kron[4*4, 84] = U_kron[16, 84]
    # V_kron[4*6, 84] = V_kron[24, 84]  
    # W_kron[6*4, 84] = W_kron[24, 84]
    
    rank = 7 * 12
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(24)]
    
    for c1 in range(7):
        for c2 in range(12):
            c = c1 * 12 + c2
            # U: index (i1*2+j1)*4 + (i2*2+j2) = i1*8 + j1*4 + i2*2 + j2
            # = (i1*2+i2)*4 + (j1*2+j2)
            for i1 in range(2):
                for j1 in range(2):
                    if U_str[i1*2+j1][c1] != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                if U_sb[i2*2+j2][c2] != 0:
                                    idx_u = (i1*2+i2)*4 + (j1*2+j2)
                                    U[idx_u][c] = U_str[i1*2+j1][c1] * U_sb[i2*2+j2][c2]
            
            # V: index (j1*2+k1)*6 + (j2*3+k2) = j1*12 + k1*6 + j2*3 + k2
            # = (j1*2+j2)*6 + (k1*3+k2)
            for j1 in range(2):
                for k1 in range(2):
                    if V_str[j1*2+k1][c1] != 0:
                        for j2 in range(2):
                            for k2 in range(3):
                                if V_sb[j2*3+k2][c2] != 0:
                                    idx_v = (j1*2+j2)*6 + (k1*3+k2)
                                    V[idx_v][c] = V_str[j1*2+k1][c1] * V_sb[j2*3+k2][c2]
            
            # W: index (k1*2+i1)*4 + (k2*2+i2) = k1*8 + i1*4 + k2*2 + i2
            # = (k1*2+k2)*4 + (i1*2+i2)
            for k1 in range(2):
                for i1 in range(2):
                    if W_str[k1*2+i1][c1] != 0:
                        for k2 in range(3):
                            for i2 in range(2):
                                if W_sb[k2*2+i2][c2] != 0:
                                    idx_w = (k1*2+k2)*4 + (i1*2+i2)
                                    W[idx_w][c] = W_str[k1*2+i1][c1] * W_sb[k2*2+i2][c2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
