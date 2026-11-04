import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    U1 = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1]
    ]
    # Wait, U1 is 2x2x7. U1[i,j,c]
    # c=0: A00+A11 -> U[0,0]=1, U[1,1]=1
    # c=1: A10+A11 -> U[1,0]=1, U[1,1]=1
    # c=2: A00 -> U[0,0]=1
    # c=3: A11 -> U[1,1]=1
    # c=4: A00+A01 -> U[0,0]=1, U[0,1]=1
    # c=5: A10-A00 -> U[1,0]=1, U[0,0]=-1
    # c=6: A01-A11 -> U[0,1]=1, U[1,1]=-1
    
    U1 = [
        [1, 0, 1, 0, 1, -1, 0], # i=0
        [0, 1, 0, 1, 0, 0, -1]  # i=1, j=0,1
    ]
    # Let's re-verify U1 indices.
    # U1[i][j] is not a matrix. It's a list of 7 values for each (i,j).
    # U1[i][j] = [u_c for c in range(7)]
    
    U1 = [
        [[1, 0, 1, 0, 1, -1, 0], [0, 0, 0, 0, 1, 0, 1]], # i=0: j=0, j=1
        [[0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0, -1]]  # i=1: j=0, j=1
    ]
    
    V1 = [
        [[1, 1, 0, -1, 0, 1, 0], [0, 0, 0, 1, 0, 1, 0]], # j=0: k=0, k=1
        [[0, 0, 1, 0, 1, 0, 1], [1, 0, -1, 0, 0, 0, 1]]  # j=1: k=0, k=1
    ]
    
    W1 = [
        [[1, 0, 0, 1, -1, 0, -1], [0, 1, 0, 0, 0, 0, 0]], # k=0: i=0, i=1
        [[0, 0, 1, 0, 1, 0, 0], [1, -1, 1, 0, 0, 1, 1]]  # k=1: i=0, i=1
    ]
    # Re-verify W1
    # c=0: W[0,0]=1, W[1,1]=1
    # c=1: W[0,1]=1, W[1,1]=-1
    # c=2: W[1,0]=1, W[1,1]=1
    # c=3: W[0,0]=1, W[0,1]=1
    # c=4: W[0,0]=-1, W[1,0]=1
    # c=5: W[0,1]=-1, W[1,1]=1
    # c=6: W[0,0]=-1, W[1,1]=1
    
    W1 = [
        [[1, 0, 0, 1, -1, 0, -1], [0, 1, 0, 1, 0, -1, 0]], # k=0: i=0, i=1
        [[0, 0, 1, 0, 1, 0, 0], [1, -1, 1, 0, 0, 1, 1]]    # k=1: i=0, i=1
    ]
    
    # Construct 2x2x4 decomposition (V2, W2, U2)
    # U2[i2][j2][c2] = U1[i2][j2][c2 % 7]
    # V2[j2][k][c2]
    # W2[k][i2][c2]
    
    for c1 in range(7):
        for c2 in range(14):
            c = 14 * c1 + c2
            c2_mod = c2 % 7
            block = c2 // 7
            
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            val_U = U1[i1][j1][c1] * U1[i2][j2][c2_mod]
                            if val_U != 0:
                                U[i * m + j][c] = val_U
                            
                            for k1 in range(2):
                                for k2 in range(4):
                                    k = 2 * k1 + k2
                                    
                                    # V2
                                    if block == 0:
                                        if k2 < 2:
                                            val_V = V1[j1][k1][c1] * V1[j2][k2][c2_mod]
                                        else:
                                            val_V = 0
                                    else:
                                        if k2 >= 2:
                                            val_V = V1[j1][k1][c1] * V1[j2][k2 - 2][c2_mod]
                                        else:
                                            val_V = 0
                                    
                                    if val_V != 0:
                                        V[j * p + k][c] = val_V
                                        
                                    # W2
                                    if block == 0:
                                        if k2 < 2:
                                            val_W = W1[k1][i1][c1] * W1[k2][i2][c2_mod]
                                        else:
                                            val_W = 0
                                    else:
                                        if k2 >= 2:
                                            val_W = W1[k1][i1][c1] * W1[k2 - 2][i2][c2_mod]
                                        else:
                                            val_W = 0
                                    
                                    if val_W != 0:
                                        W[k * n + i][c] = val_W

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
