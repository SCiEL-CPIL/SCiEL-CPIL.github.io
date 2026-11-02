"""Block-Strassen hybrid decomposition for 3x5x6 matrix multiplication."""
DIMENSIONS = [3, 5, 6]

def _strassen_2x2_2x2():
    """Return U,V,W for 2x2x2 Strassen decomposition, rank 7."""
    # Strassen: 7 multiplications
    # For 2x2x2: n=2,m=2,p=2
    # U is (2*2, 7) = (4,7), V is (2*2,7)=(4,7), W is (2*2,7)=(4,7)
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    # C00: i=0,k=0 -> W[0*2+0]=W[0]
    # C11: i=1,k=1 -> W[1*2+1]=W[3]
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    # C10 += M2
    # C10: i=1,k=0 -> W[0*2+1]=W[1]
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1  # B00
    W[1][1] = 1
    
    # M3 = A00(B10-B11)
    # C01 += M3, C11 += M3
    # C01: i=0,k=1 -> W[1*2+0]=W[2]
    # C11: i=1,k=1 -> W[1*2+1]=W[3]
    U[0][2] = 1  # A00
    V[2][2] = 1; V[3][2] = -1  # B10, -B11
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B10)
    # C00 += M4, C10 += M4
    # C00: i=0,k=0 -> W[0*2+0]=W[0]
    # C10: i=1,k=0 -> W[0*2+1]=W[1]
    U[3][3] = 1  # A11
    V[1][3] = 1; V[2][3] = -1  # B01, -B10
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    # C00 -= M5, C11 += M5
    # C00: i=0,k=0 -> W[0*2+0]=W[0]
    # C11: i=1,k=1 -> W[1*2+1]=W[3]
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1  # B11
    W[0][4] = -1; W[3][4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    # C11 += M6
    # C11: i=1,k=1 -> W[1*2+1]=W[3]
    U[2][5] = 1; U[0][5] = -1  # A10, -A00
    V[0][5] = 1; V[2][5] = 1  # B00, B10
    W[3][5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    # C00 += M7
    # C00: i=0,k=0 -> W[0*2+0]=W[0]
    U[1][6] = 1; U[3][6] = -1  # A01, -A11
    V[1][6] = 1; V[3][6] = 1  # B01, B11
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_block(nb, mb, pb):
    """Return U,V,W for nb x mb x pb schoolbook decomposition."""
    rank = nb * mb * pb
    U = [[0]*rank for _ in range(nb*mb)]
    V = [[0]*rank for _ in range(mb*pb)]
    W = [[0]*rank for _ in range(pb*nb)]
    col = 0
    for i in range(nb):
        for j in range(mb):
            for k in range(pb):
                U[i*mb+j][col] = 1
                V[j*pb+k][col] = 1
                W[k*nb+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Block splits
    n_splits = [1, 1, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for ni, nb in enumerate(n_splits):
        for mi, mb in enumerate(m_splits):
            for pi, pb in enumerate(p_splits):
                if nb == 2 and mb == 2 and pb == 2:
                    ub, vb, wb = _strassen_2x2_2x2()
                else:
                    ub, vb, wb = _schoolbook_block(nb, mb, pb)
                U_blocks.append((ni, nb, ub))
                V_blocks.append((mi, mb, vb))
                W_blocks.append((pi, pb, wb))
    
    # Now assemble full matrices
    # U: rows indexed by i*m+j for i in [0,n), j in [0,m)
    # V: rows indexed by j*p+k for j in [0,m), k in [0,p)
    # W: rows indexed by k*n+i for k in [0,p), i in [0,n)
    
    total_rank = sum(len(ub[0]) for _, _, ub in U_blocks)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for (ni, nb, ub), (mi, mb, vb), (pi, pb, wb) in zip(U_blocks, V_blocks, W_blocks):
        r = len(ub[0])
        # Compute row offsets
        n_off = sum(n_splits[:ni])
        m_off = sum(m_splits[:mi])
        p_off = sum(p_splits[:pi])
        
        for ri in range(nb):
            for rj in range(mb):
                for c in range(r):
                    U[(n_off+ri)*m + (m_off+rj)][col_offset+c] = ub[ri*mb+rj][c]
        
        for rj in range(mb):
            for rk in range(pb):
                for c in range(r):
                    V[(m_off+rj)*p + (p_off+rk)][col_offset+c] = vb[rj*pb+rk][c]
        
        for rk in range(pb):
            for ri in range(nb):
                for c in range(r):
                    W[(p_off+rk)*n + (n_off+ri)][col_offset+c] = wb[rk*nb+ri][c]
        
        col_offset += r
    
    return {"U": U, "V": V, "W": W}
