"""Strassen-based tensor decomposition for 4x4 x 4x8 matrix multiplication."""
DIMENSIONS = [4, 4, 8]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2 x 2x2 matrix multiplication, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: W[k*2+i, r] for output C[i,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    Actually for 2x2x2: U has shape (n*m, rank) = (4, 7), V has shape (m*p, rank) = (4, 7), W has shape (p*n, rank) = (4, 7)
    C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*2+j, r] V[j*2+k, r] W[k*2+i, r]
    """
    # Strassen's 7 multiplications for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # U[i*m+j, r] where i,j in {0,1}, m=2, so row = i*2+j
    # V[j*p+k, r] where j,k in {0,1}, p=2, so row = j*2+k
    # W[k*n+i, r] where k,i in {0,1}, n=2, so row = k*2+i
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = (a+d)(e+h)
    # U rows: a is (0,0)=row0, d is (1,1)=row3
    U[0][0] = 1  # a
    U[3][0] = 1  # d
    # V rows: e is (0,0)=row0, h is (1,1)=row3
    V[0][0] = 1  # e
    V[3][0] = 1  # h
    # W: M1 contributes to C[0,0] (k=0,i=0, row=0) with +1, C[1,1] (k=1,i=1, row=3) with +1
    W[0][0] = 1  # C[0,0]
    W[3][0] = 1  # C[1,1]
    
    # r=1: M2 = (c+d)e
    U[2][1] = 1  # c (1,0) -> row 1*2+0=2
    U[3][1] = 1  # d
    V[0][1] = 1  # e
    # W: M2 contributes to C[1,0] (k=0,i=1, row=0*2+1=1) with +1, C[1,1] with -1
    W[1][1] = 1  # C[1,0]
    W[3][1] = -1 # C[1,1]
    
    # r=2: M3 = a(f-h)
    U[0][2] = 1  # a
    V[1][2] = 1  # f (0,1) -> row 0*2+1=1
    V[3][2] = -1 # h
    # W: M3 contributes to C[0,1] (k=1,i=0, row=1*2+0=2) with +1, C[1,1] with +1
    W[2][2] = 1  # C[0,1]
    W[3][2] = 1  # C[1,1]
    
    # r=3: M4 = d(g-e)
    U[3][3] = 1  # d
    V[2][3] = 1  # g (1,0) -> row 1*2+0=2
    V[0][3] = -1 # e
    # W: M4 contributes to C[0,0] with +1, C[1,0] with +1
    W[0][3] = 1  # C[0,0]
    W[1][3] = 1  # C[1,0]
    
    # r=4: M5 = (a+b)h
    U[0][4] = 1  # a
    U[1][4] = 1  # b (0,1) -> row 0*2+1=1
    V[3][4] = 1  # h
    # W: M5 contributes to C[0,0] with -1, C[0,1] with +1
    W[0][4] = -1 # C[0,0]
    W[2][4] = 1  # C[0,1]
    
    # r=5: M6 = (c-a)(e+f)
    U[2][5] = 1  # c
    U[0][5] = -1 # a
    V[0][5] = 1  # e
    V[1][5] = 1  # f
    # W: M6 contributes to C[1,1] with +1
    W[3][5] = 1  # C[1,1]
    
    # r=6: M7 = (b-d)(g+h)
    U[1][6] = 1  # b
    U[3][6] = -1 # d
    V[2][6] = 1  # g
    V[3][6] = 1  # h
    # W: M7 contributes to C[0,0] with +1
    W[0][6] = 1  # C[0,0]
    
    return U, V, W

def _kron_decomp_4x4x4():
    """Return (U, V, W) for 4x4 x 4x4 matrix multiplication, rank 49.
    Uses Kronecker product of two Strassen decompositions.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # For 4x4x4, we view A as 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # Similarly B and C
    # The Kronecker product approach: for each pair of Strassen indices (r1, r2),
    # we get a term in the 4x4x4 decomposition
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # For each Strassen term (r1, r2), the contribution to U, V, W
            # is the Kronecker product of the corresponding entries
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1*2 + j1][r1]
                    if u1 != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                u2 = U2[i2*2 + j2][r2]
                                if u2 != 0:
                                    # Row index in 4x4: i = i1*2 + i2, j = j1*2 + j2
                                    row_u = i1 * 4 + j1  # Wait, need to be careful with indexing
                                    # Actually for 4x4: U[i*4+j, r]
                                    # The block structure: A[i,j] where i,j in 0..3
                                    # i = i1*2 + i2, j = j1*2 + j2
                                    # So row = i*4 + j = (i1*2+i2)*4 + (j1*2+j2)
                                    row_u = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                                    U4[row_u][r] += u1 * u2
                                    
                                    for j1v in range(2):
                                        for k1v in range(2):
                                            v1 = V2[j1v*2 + k1v][r1]
                                            if v1 != 0:
                                                for j2v in range(2):
                                                    for k2v in range(2):
                                                        v2 = V2[j2v*2 + k2v][r2]
                                                        if v2 != 0:
                                                            # V row: j*4+k where j = j1v*2+j2v, k = k1v*2+k2v
                                                            row_v = (j1v * 2 + j2v) * 4 + (k1v * 2 + k2v)
                                                            V4[row_v][r] += v1 * v2
                                                            
                                                            for k1w in range(2):
                                                                for i1w in range(2):
                                                                    w1 = W2[k1w*2 + i1w][r1]
                                                                    if w1 != 0:
                                                                        for k2w in range(2):
                                                                            for i2w in range(2):
                                                                                w2 = W2[k2w*2 + i2w][r2]
                                                                                if w2 != 0:
                                                                                    # W row: k*4+i where k = k1w*2+k2w, i = i1w*2+i2w
                                                                                    row_w = (k1w * 2 + k2w) * 4 + (i1w * 2 + i2w)
                                                                                    W4[row_w][r] += w1 * w2
    
    return U4, V4, W4

def construct():
    """Construct tensor decomposition for 4x4 x 4x8 matrix multiplication.
    Split B (4x8) into B1 (4x4) and B2 (4x4).
    C = A*B = [A*B1, A*B2]
    Each A*Bk is 4x4x4 with rank 49 using Strassen Kronecker.
    Total rank = 49 + 49 = 98.
    """
    import time
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit
    
    U4, V4, W4 = _kron_decomp_4x4x4()
    
    # For 4x4 x 4x8:
    # A is 4x4, B is 4x8, C is 4x8
    # Split B = [B1, B2] where B1, B2 are 4x4
    # C = [A*B1, A*B2]
    # 
    # U: (n*m, rank) = (16, 98)
    # V: (m*p, rank) = (32, 98) 
    # W: (p*n, rank) = (32, 98)
    #
    # For the first half (B1): 
    #   U_first[i*4+j, r] = U4[i*4+j, r] for r in 0..48
    #   V_first[j*8+k, r] = V4[j*4+k, r] for j in 0..3, k in 0..3 (k indexes into B1)
    #   W_first[k*4+i, r] = W4[k*4+i, r] for k in 0..3 (k indexes into C's first 4 columns)
    #
    # For the second half (B2):
    #   U_second[i*4+j, r] = U4[i*4+j, r-49] for r in 49..97
    #   V_second[j*8+k, r] = V4[j*4+(k-4), r-49] for j in 0..3, k in 4..7
    #   W_second[k*4+i, r] = W4[(k-4)*4+i, r-49] for k in 4..7
    
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # First half: B1 (columns 0-3 of B)
    for r in range(49):
        for row_u in range(16):
            U[row_u][r] = U4[row_u][r]
        for row_v in range(16):
            # V row for B1: j*8 + k where k in 0..3
            j = row_v // 4
            k = row_v % 4
            V[j * 8 + k][r] = V4[row_v][r]
        for row_w in range(16):
            # W row for C1: k*4 + i where k in 0..3
            k = row_w // 4
            i = row_w % 4
            W[k * 4 + i][r] = W4[row_w][r]
    
    # Second half: B2 (columns 4-7 of B)
    for r in range(49):
        r2 = r + 49
        for row_u in range(16):
            U[row_u][r2] = U4[row_u][r]
        for row_v in range(16):
            # V row for B2: j*8 + (k+4) where k in 0..3
            j = row_v // 4
            k = row_v % 4
            V[j * 8 + (k + 4)][r2] = V4[row_v][r]
        for row_w in range(16):
            # W row for C2: (k+4)*4 + i where k in 0..3
            k = row_w // 4
            i = row_w % 4
            W[(k + 4) * 4 + i][r2] = W4[row_w][r]
    
    return {"U": U, "V": V, "W": W}
