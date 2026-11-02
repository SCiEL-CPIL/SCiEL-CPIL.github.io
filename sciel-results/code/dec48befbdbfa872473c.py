import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    n, m, p = DIMENSIONS
    
    # Total rank estimate
    # M143: 3
    # M144: 4
    # M243: 18
    # M244: 28
    # Total: 53
    
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to add a sub-decomposition
    def add_sub(n_sub, m_sub, p_sub, sub_U, sub_V, sub_W, i_off, j_off, k_off):
        nonlocal current_rank
        r_sub = len(sub_U[0])
        for i in range(n_sub):
            for j in range(m_sub):
                u_idx = (i_off + i) * m + (j_off + j)
                for r in range(r_sub):
                    U[u_idx][current_rank + r] += sub_U[i * m_sub + j][r]
        for j in range(m_sub):
            for k in range(p_sub):
                v_idx = (j_off + j) * p + (k_off + k)
                for r in range(r_sub):
                    V[v_idx][current_rank + r] += sub_V[j * p_sub + k][r]
        for k in range(p_sub):
            for i in range(n_sub):
                w_idx = (k_off + k) * n + (i_off + i)
                for r in range(r_sub):
                    W[w_idx][current_rank + r] += sub_W[k * n_sub + i][r]
        current_rank += r_sub
        
    # 1. M143 (1x4 * 4x3) -> Rank 3
    U1, V1, W1 = gen_M143()
    add_sub(1, 4, 3, U1, V1, W1, 0, 0, 0)
    
    # 2. M144 (1x4 * 4x4) -> Rank 4
    U2, V2, W2 = gen_M144()
    add_sub(1, 4, 4, U2, V2, W2, 0, 0, 3)
    
    # 3. M243 (2x4 * 4x3) -> Rank 18
    U3, V3, W3 = gen_M243()
    add_sub(2, 4, 3, U3, V3, W3, 1, 0, 0)
    
    # 4. M244 (2x4 * 4x4) -> Rank 28
    U4, V4, W4 = gen_M244()
    add_sub(2, 4, 4, U4, V4, W4, 1, 0, 3)
    
    return {"U": U, "V": V, "W": W}

def gen_M143():
    # 1x4 * 4x3. 3 dot products.
    n, m, p = 1, 4, 3
    rank = 3
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for k in range(p):
        for j in range(m):
            U[0*m+j][k] = 1
            V[j*p+k][k] = 1
        W[k*n+0][k] = 1
    return U, V, W

def gen_M144():
    # 1x4 * 4x4. 4 dot products.
    n, m, p = 1, 4, 4
    rank = 4
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for k in range(p):
        for j in range(m):
            U[0*m+j][k] = 1
            V[j*p+k][k] = 1
        W[k*n+0][k] = 1
    return U, V, W

def gen_M222_strassen():
    # 2x2 * 2x2. Strassen Rank 7.
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # M1 = (A11 - A22)(B22 - B11)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A22)(B11 + B22)
    # M6 = (A12 - A11)(B11 + B21)
    # M7 = (A21 - A22)(B12 + B22)
    
    # Indices: i in 0,1; j in 0,1; k in 0,1
    # A11: i0,j0; A12: i0,j1; A21: i1,j0; A22: i1,j1
    # B11: j0,k0; B12: j0,k1; B21: j1,k0; B22: j1,k1
    
    # We define the linear forms for U (A), V (B), W (C)
    # C11 = M5 + M4 - M2 - M7
    # C12 = M1 + M2
    # C21 = M3 + M4
    # C22 = M5 - M1 + M3 + M7
    
    # U rows: 0:00, 1:01, 2:10, 3:11
    # V rows: 0:00, 1:01, 2:10, 3:11
    # W rows: 0:00, 1:01, 2:10, 3:11 (k*n+i)
    
    # M1
    U[0][0] = 1; U[3][0] = -1
    V[3][0] = 1; V[0][0] = -1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = -1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] = -1; W[2][1] = -1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[2][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1; W[2][4] = 1; W[3][4] = -1
    
    # M6 (Not used in C? Wait. Strassen C formulas:
    # C11 = M5 + M4 - M2 - M7
    # C12 = M1 + M2
    # C21 = M3 + M4
    # C22 = M5 - M1 + M3 + M7
    # M6 is not in C?
    # Ah, M6 is used in some formulations, but standard Strassen:
    # M1 = (A11 - A22)(B22 - B11)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A22)(B11 + B22)
    # M6 = (A12 - A11)(B11 + B21)
    # M7 = (A21 - A22)(B12 + B22)
    
    # C11 = M5 + M4 - M2 - M7
    # C12 = M1 + M2
    # C21 = M3 + M4
    # C22 = M5 - M1 + M3 + M7
    
    # M6 is not used?
    # Let me check.
    # C12 = A11 B12 + A12 B22.
    # M1 + M2 = (A11 - A22)(B22 - B11) + (A21 + A22)B11
    # = A11 B22 - A11 B11 - A22 B22 + A22 B11 + A21 B11 + A22 B11
    # This doesn't look like C12.
    
    # Correct Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = (A11 - A22)(B21 + B22)
    # M4 = A11(B12 - B22)
    # M5 = A22(B21 - B11)
    # M6 = (A12 + A11)B21
    # M7 = (A12 - A22)(B11 + B12)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M2 + M3
    # C21 = M6 - M1 + M4 + M5
    # C22 = M1 - M3 + M6 - M2
    
    # Let's re-implement with this standard set.
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A11 + A22)(B11 + B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = -1
    
    # M2: (A21 + A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: (A11 - A22)(B21 + B22)
    U[0][2] = 1; U[3][2] = -1
    V[2][2] = 1; V[3][2] = 1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A11(B12 - B22)
    U[0][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: A22(B21 - B11)
    U[3][4] = 1
    V[2][4] = 1; V[0][4] = -1
    W[0][4] = -1; W[2][4] = 1
    
    # M6: (A12 + A11)B21
    U[0][5] = 1; U[1][5] = 1
    V[2][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7: (A12 - A22)(B11 + B12)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[1][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def gen_M223():
    # 2x2 * 2x3. Split p=3 into 1+2.
    # Part 1: 2x2 * 2x1 (Rank 2)
    # Part 2: 2x2 * 2x2 (Rank 7)
    n, m, p = 2, 2, 3
    rank = 9
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Part 1: k=0
    U1, V1, W1 = gen_M221()
    for i in range(4):
        for r in range(2):
            U[i][r] = U1[i][r]
    for j in range(2):
        for k in range(1):
            for r in range(2):
                V[j*3+k][r] = V1[j*1+k][r]
    for k in range(1):
        for i in range(2):
            for r in range(2):
                W[k*2+i][r] = W1[k*2+i][r]
                
    # Part 2: k=1,2 (locally k=0,1)
    U2, V2, W2 = gen_M222_strassen()
    for i in range(4):
        for r in range(7):
            U[i][2+r] = U2[i][r]
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V[j*3 + (1+k)][2+r] = V2[j*2+k][r]
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W[(1+k)*2+i][2+r] = W2[k*2+i][r]
                
    return U, V, W

def gen_M221():
    # 2x2 * 2x1. Rank 2.
    n, m, p = 2, 2, 1
    rank = 2
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for j in range(2):
        for i in range(2):
            U[i*2+j][j] = 1
            V[j*1+0][j] = 1
            W[0*2+i][j] = 1
    return U, V, W

def gen_M243():
    # 2x4 * 4x3. Split m=4 into 2+2.
    # Part 1: 2x2 * 2x3 (Rank 9)
    # Part 2: 2x2 * 2x3 (Rank 9)
    n, m, p = 2, 4, 3
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U1, V1, W1 = gen_M223()
    for i in range(2):
        for j in range(2):
            for r in range(9):
                U[i*4+j][r] = U1[i*2+j][r]
    for j in range(2):
        for k in range(3):
            for r in range(9):
                V[j*3+k][r] = V1[j*3+k][r]
    for k in range(3):
        for i in range(2):
            for r in range(9):
                W[k*2+i][r] = W1[k*2+i][r]
                
    U2, V2, W2 = gen_M223()
    for i in range(2):
        for j in range(2):
            for r in range(9):
                U[i*4 + (2+j)][9+r] = U2[i*2+j][r]
    for j in range(2):
        for k in range(3):
            for r in range(9):
                V[(2+j)*3+k][9+r] = V2[j*3+k][r]
    for k in range(3):
        for i in range(2):
            for r in range(9):
                W[k*2+i][9+r] = W2[k*2+i][r]
                
    return U, V, W

def gen_M224():
    # 2x2 * 2x4. Split p=4 into 2+2.
    # Part 1: 2x2 * 2x2 (Rank 7)
    # Part 2: 2x2 * 2x2 (Rank 7)
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U1, V1, W1 = gen_M222_strassen()
    for i in range(4):
        for r in range(7):
            U[i][r] = U1[i][r]
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V[j*4+k][r] = V1[j*2+k][r]
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W[k*2+i][r] = W1[k*2+i][r]
                
    U2, V2, W2 = gen_M222_strassen()
    for i in range(4):
        for r in range(7):
            U[i][7+r] = U2[i][r]
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V[j*4 + (2+k)][7+r] = V2[j*2+k][r]
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W[(2+k)*2+i][7+r] = W2[k*2+i][r]
                
    return U, V, W

def gen_M244():
    # 2x4 * 4x4. Split m=4 into 2+2.
    # Part 1: 2x2 * 2x4 (Rank 14)
    # Part 2: 2x2 * 2x4 (Rank 14)
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U1, V1, W1 = gen_M224()
    for i in range(2):
        for j in range(2):
            for r in range(14):
                U[i*4+j][r] = U1[i*2+j][r]
    for j in range(2):
        for k in range(4):
            for r in range(14):
                V[j*4+k][r] = V1[j*4+k][r]
    for k in range(4):
        for i in range(2):
            for r in range(14):
                W[k*2+i][r] = W1[k*2+i][r]
                
    U2, V2, W2 = gen_M224()
    for i in range(2):
        for j in range(2):
            for r in range(14):
                U[i*4 + (2+j)][14+r] = U2[i*2+j][r]
    for j in range(2):
        for k in range(4):
            for r in range(14):
                V[(2+j)*4+k][14+r] = V2[j*4+k][r]
    for k in range(4):
        for i in range(2):
            for r in range(14):
                W[k*2+i][14+r] = W2[k*2+i][r]
                
    return U, V, W

