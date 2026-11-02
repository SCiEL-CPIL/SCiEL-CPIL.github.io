DIMENSIONS = [4, 4, 7]

def strassen_2x2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[2][2] = 1
    W[3][2] = -1
    
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1
    W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[0][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[1][6] = 1
    W[3][6] = 1
    
    return U, V, W

def expand_strassen_4x4x4():
    U2, V2, W2 = strassen_2x2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for c_outer in range(7):
        for sc_inner in range(7):
            col = c_outer * 7 + sc_inner
            for i_outer in range(2):
                for j_outer in range(2):
                    for i_inner in range(2):
                        for j_inner in range(2):
                            u_val = U2[i_outer*2+j_outer][c_outer] * U2[i_inner*2+j_inner][sc_inner]
                            if u_val != 0:
                                row_u = (i_outer*2 + i_inner)*4 + (j_outer*2 + j_inner)
                                U[row_u][col] = u_val
                                
            for j_outer in range(2):
                for k_outer in range(2):
                    for j_inner in range(2):
                        for k_inner in range(2):
                            v_val = V2[j_outer*2+k_outer][c_outer] * V2[j_inner*2+k_inner][sc_inner]
                            if v_val != 0:
                                row_v = (j_outer*2 + j_inner)*4 + (k_outer*2 + k_inner)
                                V[row_v][col] = v_val
                                
            for k_outer in range(2):
                for i_outer in range(2):
                    for k_inner in range(2):
                        for i_inner in range(2):
                            w_val = W2[k_outer*2+i_outer][c_outer] * W2[k_inner*2+i_inner][sc_inner]
                            if w_val != 0:
                                row_w = (k_outer*2 + k_inner)*4 + (i_outer*2 + i_inner)
                                W[row_w][col] = w_val
                                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U1, V1, W1 = expand_strassen_4x4x4()
    
    rank2 = 4 * 4 * 3
    U2 = [[0]*rank2 for _ in range(16)]
    V2 = [[0]*rank2 for _ in range(12)]
    W2 = [[0]*rank2 for _ in range(12)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U2[i*4+j][col] = 1
                V2[j*3+k][col] = 1
                W2[k*4+i][col] = 1
                col += 1
                
    total_rank = 49 + rank2
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    for r in range(16):
        for c in range(49):
            U[r][c] = U1[r][c]
            V[r][c] = V1[r][c]
            W[r][c] = W1[r][c]
            
    for r in range(16):
        for c in range(rank2):
            U[r][49+c] = U2[r][c]
            
    for r in range(12):
        for c in range(rank2):
            j = r // 3
            k = r % 3
            v_row = j * 7 + (k + 4)
            V[v_row][49+c] = V2[r][c]
            
    for r in range(12):
        for c in range(rank2):
            k = r // 4
            i = r % 4
            w_row = (k + 4) * 4 + i
            W[w_row][49+c] = W2[r][c]
            
    return {"U": U, "V": V, "W": W}
