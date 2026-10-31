"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 49
    rank = 49
    
    # Define 2x2 Strassen matrices
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    # M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    # M3
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    # M4
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    # M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    # M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[2][5] = 1
    W2[3][5] = 1
    # M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[3][6] = 1; V2[1][6] = 1
    W2[0][6] = 1

    # Construct 4x4 matrices
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            I_outer = i // 2
            i_inner = i % 2
            J_outer = j // 2
            j_inner = j % 2
            row_u = i * m + j
            
            for c_outer in range(7):
                val_u_outer = U2[I_outer*2 + J_outer][c_outer]
                if val_u_outer == 0: continue
                for c_inner in range(7):
                    val_u_inner = U2[i_inner*2 + j_inner][c_inner]
                    if val_u_inner == 0: continue
                    c = c_outer * 7 + c_inner
                    U[row_u][c] += val_u_outer * val_u_inner
                    
    for j in range(m):
        for k in range(p):
            J_outer = j // 2
            j_inner = j % 2
            K_outer = k // 2
            k_inner = k % 2
            row_v = j * p + k
            
            for c_outer in range(7):
                val_v_outer = V2[J_outer*2 + K_outer][c_outer]
                if val_v_outer == 0: continue
                for c_inner in range(7):
                    val_v_inner = V2[j_inner*2 + k_inner][c_inner]
                    if val_v_inner == 0: continue
                    c = c_outer * 7 + c_inner
                    V[row_v][c] += val_v_outer * val_v_inner
                    
    for k in range(p):
        for i in range(n):
            K_outer = k // 2
            k_inner = k % 2
            I_outer = i // 2
            i_inner = i % 2
            row_w = k * n + i
            
            for c_outer in range(7):
                val_w_outer = W2[K_outer*2 + I_outer][c_outer]
                if val_w_outer == 0: continue
                for c_inner in range(7):
                    val_w_inner = W2[k_inner*2 + i_inner][c_inner]
                    if val_w_inner == 0: continue
                    c = c_outer * 7 + c_inner
                    W[row_w][c] += val_w_outer * val_w_inner

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
