import time

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 -> 2x2 multiplication.
    U, V, W are lists of lists.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    Rank 7.
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (a00+a11)*(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)*b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a00*(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a11*(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # M5 = (a00+a01)*b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # M6 = (a10-a00)*(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7 = (a01-a11)*(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _build_4x4x4():
    """
    Builds decomposition for 4x4 * 4x4 -> 4x4 using Strassen blocks.
    Returns U (16x49), V (16x49), W (16x49).
    Indexing: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c].
    """
    n = 4
    rank = 49
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    
    sU, sV, sW = _strassen_2x2()
    
    # Block multipliers for Strassen 2x2 blocks
    # A blocks: 0=A11, 1=A12, 2=A21, 3=A22
    # B blocks: 0=B11, 1=B12, 2=B21, 3=B22
    # C blocks: 0=C11, 1=C12, 2=C21, 3=C22
    
    mults = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),    # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),          # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),          # M3
        ({3:1}, {2:1, 0:-1}, {2:1, 0:1}),          # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 3:1}),          # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {2:1, 3:1}),     # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:-1, 3:1}),    # M7
    ]
    
    for mi, (a_c, b_c, c_c) in enumerate(mults):
        for s in range(7):
            col = mi * 7 + s
            for xi in range(4):
                xr = xi // 2
                xc = xi % 2
                u_val = sU[xi][s]
                if u_val != 0:
                    for ab, coeff in a_c.items():
                        global_u = ab * 4 + xr * 2 + xc
                        U[global_u][col] += u_val * coeff
            
            for yi in range(4):
                yr = yi // 2
                yc = yi % 2
                v_val = sV[yi][s]
                if v_val != 0:
                    for bb, coeff in b_c.items():
                        global_v = bb * 4 + yr * 2 + yc
                        V[global_v][col] += v_val * coeff
            
            for zi in range(4):
                zr = zi // 2
                zc = zi % 2
                w_val = sW[zi][s]
                if w_val != 0:
                    for cb, coeff in c_c.items():
                        global_w = cb * 4 + zr * 2 + zc
                        W[global_w][col] += w_val * coeff
                        
    return U, V, W

def _build_4x4x1():
    """
    Builds decomposition for 4x4 * 4x1 -> 4x1.
    Returns U (16x16), V (4x16), W (4x16).
    Indexing: U[i*4+j, c], V[j*1+k, c] (k=0), W[k*4+i, c] (k=0).
    Note: V index is j*p+k = j*1+0 = j. W index is k*n+i = 0*4+i = i.
    """
    n = 4
    m = 4
    p = 1
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    return U, V, W

def construct():
    n, m, p = 4, 4, 5
    deadline = time.time() + 100 # Safety margin
    
    # Build 4x4x4 part (rank 49)
    U_s, V_s, W_s = _build_4x4x4()
    
    # Build 4x4x1 part (rank 16)
    U_e, V_e, W_e = _build_4x4x1()
    
    total_rank = 49 + 16
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Copy Strassen part
    # U: 16x49 -> 16x65 (cols 0-48)
    for i in range(16):
        for c in range(49):
            U[i][c] = U_s[i][c]
            
    # V: 16x49 -> 20x65 (cols 0-48, rows 0-15)
    # V_s is for 4x4x4, so V_s[j*4+k].
    # In 4x4x5, V[j*5+k].
    # We map V_s[j*4+k] to V[j*5+k] for k=0..3.
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*5+k][c] = V_s[j*4+k][c]
                
    # W: 16x49 -> 20x65 (cols 0-48, rows 0-15)
    # W_s is for 4x4x4, so W_s[k*4+i].
    # In 4x4x5, W[k*5+i].
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*5+i][c] = W_s[k*4+i][c]
                
    # Copy Extra part
    # U_e: 16x16 -> 16x65 (cols 49-64)
    for i in range(16):
        for c in range(16):
            U[i][49+c] = U_e[i][c]
            
    # V_e: 4x16 -> 20x65 (cols 49-64, row 16-19)
    # V_e is for 4x4x1, so V_e[j*1+0] = V_e[j].
    # In 4x4x5, V[j*5+4].
    for j in range(4):
        for c in range(16):
            V[j*5+4][49+c] = V_e[j][c]
            
    # W_e: 4x16 -> 20x65 (cols 49-64, row 16-19)
    # W_e is for 4x4x1, so W_e[0*4+i] = W_e[i].
    # In 4x4x5, W[4*5+i] = W[20+i]? No, k=4. W[4*4+i] = W[16+i].
    # Wait, W index is k*n+i. n=4. k=4. Index 16+i.
    for i in range(4):
        for c in range(16):
            W[4*4+i][49+c] = W_e[i][c]
            
    return {"U": U, "V": V, "W": W}
