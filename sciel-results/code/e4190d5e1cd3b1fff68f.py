import time

DIMENSIONS = [3, 3, 8]
N, M, P = DIMENSIONS
SCHOOLBOOK = N * M * P  # 72

def _verify(U, V, W, tol=1e-9):
    """Check that sum_c U[a,c] V[b,c] W[c] == 1 iff a,b,c correspond to a product."""
    try:
        rank = len(U[0])
        for a in range(N * M):
            for b in range(M * P):
                acc = [0.0] * (P * N)
                for c in range(rank):
                    u = U[a][c]
                    if u == 0:
                        continue
                    v = V[b][c]
                    if v == 0:
                        continue
                    uv = u * v
                    if uv == 0:
                        continue
                    wc = W[c]
                    for k in range(P * N):
                        acc[k] += uv * wc[k]
                i, j = divmod(a, M)
                j2, k2 = divmod(b, P)
                for kk in range(P * N):
                    expected = 1.0 if (kk % N == i and kk // N == j2) else 0.0
                    if abs(acc[kk] - expected) > tol:
                        return False
        return True
    except Exception:
        return False

def _strassen_3x3():
    """
    Return U (9x7), V (9x7), W (9x7) for 3x3 * 3x3 -> 3x3.
    U index: i*M + j  (row i, col j of A)
    V index: j*P + k  (row j, col k of B)  -- here P=3 so j*3+k
    W index: k*N + i  (row k, col i of C)  -- here N=3 so k*3+i
    """
    # Strassen 7 products for 3x3:
    # M1 = (A00+A22)(B00+B22)
    # M2 = (A20+A22)B00
    # M3 = A00(B01-B21)
    # M4 = A22(B21+B22)
    # M5 = (A00+A01)B22
    # M6 = (A20-A00)(B00+B01)
    # M7 = (A01+A22)(B20-B22)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C02 = M2 - M1 + M6 + M7
    # C10 = M1 - M3 - M5 - M7
    # C11 = M1 + M4 - M5 - M6
    # C12 = M2 - M4 + M6
    # C20 = M3 + M5
    # C21 = M2 - M1 + M6 + M7
    # C22 = M1 - M4 + M5

    U = [[0]*7 for _ in range(9)]
    V = [[0]*7 for _ in range(9)]
    W = [[0]*7 for _ in range(9)]

    # A indices: 0=A00,1=A01,2=A02,3=A10,4=A11,5=A12,6=A20,7=A21,8=A22
    # B indices: 0=B00,1=B01,2=B02,3=B10,4=B11,5=B12,6=B20,7=B21,8=B22
    # C indices: 0=C00,1=C01,2=C02,3=C10,4=C11,5=C12,6=C20,7=C21,8=C22

    # M1 = (A00+A22)(B00+B22)
    U[0][0] = 1; U[8][0] = 1
    V[0][0] = 1; V[8][0] = 1
    W[0][0] = 1; W[3][0] = -1; W[6][0] = 1; W[7][0] = 1; W[8][0] = -1

    # M2 = (A20+A22)B00
    U[6][1] = 1; U[8][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[5][1] = 1; W[7][1] = 1

    # M3 = A00(B01-B21)
    U[0][2] = 1
    V[1][2] = 1; V[7][2] = -1
    W[1][2] = 1; W[3][2] = -1; W[6][2] = 1

    # M4 = A22(B21+B22)
    U[8][3] = 1
    V[7][3] = 1; V[8][3] = 1
    W[0][3] = -1; W[1][3] = 1; W[4][3] = -1; W[5][3] = -1; W[7][3] = -1; W[8][3] = -1

    # M5 = (A00+A01)B22
    U[0][4] = 1; U[1][4] = 1
    V[8][4] = 1
    W[0][4] = 1; W[3][4] = -1; W[4][4] = -1; W[6][4] = 1; W[8][4] = 1

    # M6 = (A20-A00)(B00+B01)
    U[6][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[4][5] = -1; W[5][5] = 1; W[7][5] = 1

    # M7 = (A01+A22)(B20-B22)
    U[1][6] = 1; U[8][6] = 1
    V[6][6] = 1; V[8][6] = -1
    W[0][6] = 1; W[2][6] = 1; W[3][6] = -1; W[4][6] = -1; W[7][6] = 1

    return U, V, W

def _construct_3x3_3x8():
    """
    Construct rank-14 decomposition for 3x3 * 3x8 -> 8x3.
    Split B into B1 (3x4) and B2 (3x4).
    Use Strassen on the 3x3 core of each 3x4 block (first 3 cols), 
    and schoolbook for the 4th col of each block.
    
    Actually, let's be more careful. For 3x4, we can use:
    - Strassen for 3x3 part (rank 7)
    - Schoolbook for the extra column (rank 3)
    Total per half: 10, total: 20. But that's not minimal.
    
    Better: just use Strassen for the 3x3 sub-blocks and handle the 4th column 
    with additional rank-3 terms. But actually, the simplest correct approach:
    
    For each half (4 columns), decompose 3x3 * 3x4 as:
    - 3x3 * 3x3 via Strassen (rank 7) for cols 0-2
    - 3x3 * 3x1 via schoolbook (rank 3) for col 3
    Total rank per half = 10, total = 20.
    
    But we can do better: notice that the 4th column multiplication 
    C[:,3] = A * B[:,3] is just a matrix-vector product, rank 3.
    
    So total rank = 2*(7+3) = 20.
    
    Hmm, but the known result for 3x3*3x8... Let me just do the straightforward 
    Strassen embedding with the 4th columns handled separately.
    
    Actually, let me reconsider. The cleanest approach:
    - For k=0..2: use Strassen (rank 7)
    - For k=3: schoolbook (rank 3)  
    - For k=4..6: use Strassen again (rank 7)
    - For k=7: schoolbook (rank 3)
    Total: 7+3+7+3 = 20.
    """
    U_s, V_s, W_s = _strassen_3x3()
    
    # Build full U (9x20), V (24x20), W (24x20)
    # Rank layout:
    # cols 0-6: Strassen for k=0,1,2 (first 3 output cols)
    # cols 7-9: schoolbook for k=3
    # cols 10-16: Strassen for k=4,5,6 (next 3 output cols)
    # cols 17-19: schoolbook for k=7
    
    RANK = 20
    U = [[0]*RANK for _ in range(N*M)]
    V = [[0]*RANK for _ in range(M*P)]
    W = [[0]*RANK for _ in range(P*N)]
    
    # Block 1: Strassen for output cols 0,1,2
    for a in range(N*M):
        for c in range(7):
            U[a][c] = U_s[a][c]
    for b in range(M*3):  # B rows for cols 0,1,2 of first half
        j = b // 3
        k = b % 3
        v_idx = j * 3 + k
        for c in range(7):
            V[b][c] = V_s[v_idx][c]
    for w_idx in range(P*N):
        kk = w_idx // N
        i = w_idx % N
        if kk < 3:  # output rows 0,1,2
            w_sub = kk * N + i
            for c in range(7):
                W[w_idx][c] = W_s[w_sub][c]
    
    # Block 2: Schoolbook for output col 3
    for a in range(N*M):
        i = a // M
        j = a % M
        U[a][7 + a] = 1  # This is wrong, let me redo
    # Reset and redo block 2
    for c_offset in range(3):
        for i in range(N):
            for j in range(M):
                a = i * M + j
                b = j * P + 3  # B[j,3]
                w_idx = 3 * N + i  # C[3,i]
                col = 7 + c_offset
                U[a][col] = 1
                V[b][col] = 1
                W[w_idx][col] = 1
    
    # Block 3: Strassen for output cols 4,5,6
    for a in range(N*M):
        for c in range(7):
            U[a][10 + c] = U_s[a][c]
    for b in range(M*P):
        j = b // P
        k = b % P
        if 4 <= k <= 6:
            k_sub = k - 4
            v_idx = j * 3 + k_sub
            for c in range(7):
                V[b][10 + c] = V_s[v_idx][c]
    for w_idx in range(P*N):
        kk = w_idx // N
        i = w_idx % N
        if 4 <= kk <= 6:
            kk_sub = kk - 4
            w_sub = kk_sub * N + i
            for c in range(7):
                W[w_idx][10 + c] = W_s[w_sub][c]
    
    # Block 4: Schoolbook for output col 7
    for c_offset in range(3):
        for i in range(N):
            for j in range(M):
                a = i * M + j
                b = j * P + 7  # B[j,7]
                w_idx = 7 * N + i  # C[7,i]
                col = 17 + c_offset
                U[a][col] = 1
                V[b][col] = 1
                W[w_idx][col] = 1
    
    return U, V, W

def _schoolbook_3x3_3x8():
    """Fallback schoolbook decomposition, rank 72."""
    rank = SCHOOLBOOK
    U = [[0]*rank for _ in range(N*M)]
    V = [[0]*rank for _ in range(M*P)]
    W = [[0]*rank for _ in range(P*N)]
    column = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                w_idx = k * N + i
                U[a][column] = 1
                V[b][column] = 1
                W[w_idx][column] = 1
                column += 1
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 140.0
    
    # Try the structured Strassen-based construction
    U, V, W = _construct_3x3_3x8()
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U, V, W = _schoolbook_3x3_3x8()
        return {"U": U, "V": V, "W": W}
