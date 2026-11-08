import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define 2x2 Strassen matrices
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: rows 0..3, cols 0..6
    # Row index for U2: i*2 + j
    # Row index for V2: j*2 + k
    # Row index for W2: k*2 + i
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # alpha=0: M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # alpha=1: M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # alpha=2: M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # alpha=3: M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # alpha=4: M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # alpha=5: M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # alpha=6: M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # Construct 4x4 matrices
    # Rank = 7 * 7 = 49
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Iterate over outer alpha and inner beta
    for alpha in range(7):
        for beta in range(7):
            t = alpha * 7 + beta
            
            # Iterate over all indices for U, V, W
            # U index: i (0..3), j (0..3)
            # i = i1*2 + i2, j = j1*2 + j2
            for i in range(4):
                i1 = i // 2
                i2 = i % 2
                for j in range(4):
                    j1 = j // 2
                    j2 = j % 2
                    row_u = i * 4 + j
                    u_val = U2[i1*2 + j1][alpha] * U2[i2*2 + j2][beta]
                    if u_val != 0:
                        U[row_u][t] = u_val
            
            # V index: j (0..3), k (0..3)
            # j = j1*2 + j2, k = k1*2 + k2
            for j in range(4):
                j1 = j // 2
                j2 = j % 2
                for k in range(4):
                    k1 = k // 2
                    k2 = k % 2
                    row_v = j * 4 + k
                    v_val = V2[j1*2 + k1][alpha] * V2[j2*2 + k2][beta]
                    if v_val != 0:
                        V[row_v][t] = v_val
            
            # W index: k (0..3), i (0..3)
            # k = k1*2 + k2, i = i1*2 + i2
            # W row index is k*n + i
            for k in range(4):
                k1 = k // 2
                k2 = k % 2
                for i in range(4):
                    i1 = i // 2
                    i2 = i % 2
                    row_w = k * 4 + i
                    w_val = W2[k1*2 + i1][alpha] * W2[k2*2 + i2][beta]
                    if w_val != 0:
                        W[row_w][t] = w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
