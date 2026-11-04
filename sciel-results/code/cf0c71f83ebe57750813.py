"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Sub-decomposition 1: 2x2x2 (Strassen, rank 7)
    n1, m1, p1 = 2, 2, 2
    r1 = 7
    U1 = [[0] * r1 for _ in range(n1 * m1)]
    V1 = [[0] * r1 for _ in range(m1 * p1)]
    W1 = [[0] * r1 for _ in range(p1 * n1)]
    
    # M1 = (A00+A11)(B00+B11)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[2][0] = 1; W1[1][0] = 1; W1[3][0] = 1
    
    # M2 = (A10+A11)B00
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3 = A00(B01-B11)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4 = A11(B10-B00)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5 = (A00+A01)B11
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # Sub-decomposition 2: 2x2x3 (Rank 11: Strassen for first 2 cols + 4 mults for 3rd col)
    n2, m2, p2 = 2, 2, 3
    r2 = 11
    U2 = [[0] * r2 for _ in range(n2 * m2)]
    V2 = [[0] * r2 for _ in range(m2 * p2)]
    W2 = [[0] * r2 for _ in range(p2 * n2)]
    
    # Columns 0-6: Same as Strassen, but B only uses cols 0,1. V2 rows for col 2 are 0.
    # W2 rows for col 2 are 0.
    # Copy U1, V1, W1 logic to U2, V2, W2 for cols 0-6
    # U2 is 4x11, U1 is 4x7.
    for i in range(4):
        for j in range(7):
            U2[i][j] = U1[i][j]
            
    # V2 is 6x11. V1 is 4x7.
    # V1 rows: j1*p1+k1 -> 0,1,2,3.
    # V2 rows: j2*p2+k2 -> 0,1,2,3,4,5.
    # j2=0: k2=0,1,2 -> 0,1,2. j2=1: k2=0,1,2 -> 3,4,5.
    # Strassen uses B cols 0,1. So k2 in {0,1}.
    # V1[0] (B00) -> V2[0] (B0,0)
    # V1[1] (B01) -> V2[1] (B0,1)
    # V1[2] (B10) -> V2[3] (B1,0)
    # V1[3] (B11) -> V2[4] (B1,1)
    v_map = [0, 1, 3, 4]
    for i in range(4):
        for j in range(7):
            V2[v_map[i]][j] = V1[i][j]
            
    # W2 is 6x11. W1 is 4x7.
    # W1 rows: k1*n1+i1 -> 0,1,2,3.
    # W2 rows: k2*n2+i2 -> 0,1,2,3,4,5.
    # Strassen uses C cols 0,1. So k2 in {0,1}.
    # W1[0] (C00) -> W2[0] (C0,0)
    # W1[1] (C10) -> W2[2] (C1,0) ? No.
    # W1 index k1*n1+i1.
    # k1=0, i1=0 -> 0. C00.
    # k1=0, i1=1 -> 1. C10.
    # k1=1, i1=0 -> 2. C01.
    # k1=1, i1=1 -> 3. C11.
    # W2 index k2*n2+i2.
    # k2=0, i2=0 -> 0. C0,0.
    # k2=0, i2=1 -> 1. C1,0.
    # k2=1, i2=0 -> 2. C0,1.
    # k2=1, i2=1 -> 3. C1,1.
    # k2=2, i2=0 -> 4. C0,2.
    # k2=2, i2=1 -> 5. C1,2.
    # So mapping is identity for 0-3.
    for i in range(4):
        for j in range(7):
            W2[i][j] = W1[i][j]
            
    # Columns 7-10: 3rd column of B
    # M7 = A00 B02. U2[0][7]=1. V2[2][7]=1 (B0,2). W2[4][7]=1 (C0,2).
    U2[0][7] = 1
    V2[2][7] = 1
    W2[4][7] = 1
    
    # M8 = A01 B12. U2[1][8]=1. V2[5][8]=1 (B1,2). W2[4][8]=1 (C0,2).
    U2[1][8] = 1
    V2[5][8] = 1
    W2[4][8] = 1
    
    # M9 = A10 B02. U2[2][9]=1. V2[2][9]=1 (B0,2). W2[5][9]=1 (C1,2).
    U2[2][9] = 1
    V2[2][9] = 1
    W2[5][9] = 1
    
    # M10 = A11 B12. U2[3][10]=1. V2[5][10]=1 (B1,2). W2[5][10]=1 (C1,2).
    U2[3][10] = 1
    V2[5][10] = 1
    W2[5][10] = 1
    
    # Combine using Kronecker product
    n, m, p = DIMENSIONS
    rank = r1 * r2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        i1 = i // n2
        i2 = i % n2
        for j in range(m):
            j1 = j // m2
            j2 = j % m2
            u_idx = i * m + j
            u1_idx = i1 * m1 + j1
            u2_idx = i2 * m2 + j2
            
            for r1_idx in range(r1):
                val1 = U1[u1_idx][r1_idx]
                if val1 == 0: continue
                for r2_idx in range(r2):
                    val2 = U2[u2_idx][r2_idx]
                    if val2 == 0: continue
                    r = r1_idx * r2 + r2_idx
                    U[u_idx][r] = val1 * val2
                    
    for j in range(m):
        j1 = j // m2
        j2 = j % m2
        for k in range(p):
            k1 = k // p2
            k2 = k % p2
            v_idx = j * p + k
            v1_idx = j1 * p1 + k1
            v2_idx = j2 * p2 + k2
            
            for r1_idx in range(r1):
                val1 = V1[v1_idx][r1_idx]
                if val1 == 0: continue
                for r2_idx in range(r2):
                    val2 = V2[v2_idx][r2_idx]
                    if val2 == 0: continue
                    r = r1_idx * r2 + r2_idx
                    V[v_idx][r] = val1 * val2
                    
    for k in range(p):
        k1 = k // p2
        k2 = k % p2
        for i in range(n):
            i1 = i // n2
            i2 = i % n2
            w_idx = k * n + i
            w1_idx = k1 * n1 + i1
            w2_idx = k2 * n2 + i2
            
            for r1_idx in range(r1):
                val1 = W1[w1_idx][r1_idx]
                if val1 == 0: continue
                for r2_idx in range(r2):
                    val2 = W2[w2_idx][r2_idx]
                    if val2 == 0: continue
                    r = r1_idx * r2 + r2_idx
                    W[w_idx][r] = val1 * val2
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
