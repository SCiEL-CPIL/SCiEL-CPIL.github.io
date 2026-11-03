"""Strassen-based decomposition for 4x4x6 matrix multiplication."""

def _strassen_2x2x2_entries():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    def ui(i,j): return i*2+j
    def vi(j,k): return j*2+k
    def wi(k,i): return k*2+i
    U[ui(0,0)][0]=1; U[ui(1,1)][0]=1
    V[vi(0,1)][0]=1; V[vi(1,0)][0]=1
    W[wi(0,0)][0]=1; W[wi(0,1)][0]=1; W[wi(1,0)][0]=-1; W[wi(1,1)][0]=1
    U[ui(1,0)][1]=1; U[ui(1,1)][1]=1
    V[vi(0,0)][1]=1
    W[wi(1,0)][1]=1; W[wi(1,1)][1]=-1
    U[ui(0,0)][2]=1
    V[vi(1,0)][2]=1; V[vi(1,1)][2]=-1
    W[wi(0,1)][2]=1; W[wi(1,0)][2]=1
    U[ui(1,1)][3]=1
    V[vi(0,0)][3]=1; V[vi(1,0)][3]=-1
    W[wi(0,0)][3]=1; W[wi(1,0)][3]=1
    U[ui(0,0)][4]=1; U[ui(0,1)][4]=1
    V[vi(1,1)][4]=1
    W[wi(0,0)][4]=-1; W[wi(0,1)][4]=1
    U[ui(1,0)][5]=1; U[ui(0,0)][5]=-1
    V[vi(0,0)][5]=1; V[vi(0,1)][5]=1
    W[wi(1,1)][5]=1
    U[ui(0,1)][6]=1; U[ui(1,1)][6]=-1
    V[vi(1,0)][6]=1; V[vi(1,1)][6]=1
    W[wi(0,0)][6]=1
    return U, V, W

def _strassen_4x4x4():
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    u2, v2, w2 = _strassen_2x2x2_entries()
    outer_U = {
        0: [(0,0,1),(1,1,1)],
        1: [(1,0,1),(1,1,1)],
        2: [(0,0,1)],
        3: [(1,1,1)],
        4: [(0,0,1),(0,1,1)],
        5: [(1,0,1),(0,0,-1)],
        6: [(0,1,1),(1,1,-1)],
    }
    outer_V = {
        0: [(0,1,1),(1,0,1)],
        1: [(0,0,1)],
        2: [(1,0,1),(1,1,-1)],
        3: [(0,0,1),(1,0,-1)],
        4: [(1,1,1)],
        5: [(0,0,1),(0,1,1)],
        6: [(1,0,1),(1,1,1)],
    }
    outer_W = {
        0: [(0,0,1),(0,1,1),(1,0,-1),(1,1,1)],
        1: [(1,0,1),(1,1,-1)],
        2: [(0,1,1),(1,0,1)],
        3: [(0,0,1),(1,0,1)],
        4: [(0,0,-1),(0,1,1)],
        5: [(1,1,1)],
        6: [(0,0,1)],
    }
    for t in range(7):
        for s in range(7):
            col = t*7 + s
            for (bi,bj,sa) in outer_U[t]:
                for (ii,jj) in [(0,0),(0,1),(1,0),(1,1)]:
                    inner_coeff = u2[ii*2+jj][s]
                    if inner_coeff != 0:
                        i = bi*2 + ii
                        j = bj*2 + jj
                        U[i*4+j][col] += sa * inner_coeff
            for (bj,bk,sb) in outer_V[t]:
                for (jj,kk) in [(0,0),(0,1),(1,0),(1,1)]:
                    inner_coeff = v2[jj*2+kk][s]
                    if inner_coeff != 0:
                        j = bj*2 + jj
                        k = bk*2 + kk
                        V[j*4+k][col] += sb * inner_coeff
            for (ck,ci,sc) in outer_W[t]:
                for (kk,ii) in [(0,0),(0,1),(1,0),(1,1)]:
                    inner_coeff = w2[kk*2+ii][s]
                    if inner_coeff != 0:
                        k = ck*2 + kk
                        i = ci*2 + ii
                        W[k*4+i][col] += sc * inner_coeff
    return U, V, W

def _strassen_4x4x2():
    U = [[0]*28 for _ in range(16)]
    V = [[0]*28 for _ in range(8)]
    W = [[0]*28 for _ in range(8)]
    u2, v2, w2 = _strassen_2x2x2_entries()
    blocks = [
        (0,0,0,0),  # A_row, A_col, B_row, C_row
        (0,1,1,0),
        (1,0,0,1),
        (1,1,1,1),
    ]
    for bi, (ar, ac, br, cr) in enumerate(blocks):
        for s in range(7):
            col = bi*7 + s
            for ii in range(2):
                for jj in range(2):
                    coeff = u2[ii*2+jj][s]
                    if coeff != 0:
                        i = ar*2 + ii
                        j = ac*2 + jj
                        U[i*4+j][col] = coeff
            for jj in range(2):
                for kk in range(2):
                    coeff = v2[jj*2+kk][s]
                    if coeff != 0:
                        j = br*2 + jj
                        k = kk
                        V[j*2+k][col] = coeff
            for kk in range(2):
                for ii in range(2):
                    coeff = w2[kk*2+ii][s]
                    if coeff != 0:
                        k = kk
                        i = cr*2 + ii
                        W[k*4+i][col] = coeff
    return U, V, W

def construct():
    n, m, p = 4, 4, 6
    U1, V1, W1 = _strassen_4x4x4()
    U2, V2, W2 = _strassen_4x4x2()
    rank = 49 + 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for i in range(n*m):
        for c in range(49):
            U[i][c] = U1[i][c]
        for c in range(28):
            U[i][49+c] = U2[i][c]
    for i in range(m*p):
        for c in range(49):
            V[i][c] = V1[i][c]
        for c in range(28):
            V[i][49+c] = V2[i][c]
    for i in range(p*n):
        for c in range(49):
            W[i][c] = W1[i][c]
        for c in range(28):
            W[i][49+c] = W2[i][c]
    return {"U": U, "V": V, "W": W}
