import time

# Deadline guard (evaluator kills after 150s wall-clock; safety margin ~120s per spec)
_DEADLINE = time.time() + 118

def construct():
    if time.time() > _DEADLINE - 2:
        # Fallback: return schoolbook rank 216 immediately if we're out of time
        n = m = p = 6
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*(m*p) + j*p + k
                    U[i*m + j][r] = 1
                    V[j*p + k][r] = 1
                    W[k*n + i][r] = 1
        return {"U": U, "V": V, "W": W}

    n, m, p = 6, 6, 6

    # Strassen-based block decomposition for 6x6x6
    # Treat each 6x6 as a 3x3 block matrix of 2x2 blocks.
    # Each 2x2 block multiplication uses 7 Strassen sub-multiplications.
    # Total rank = 3 * 3 * 7 = 189

    rank = 189
    U = [[0]*rank for _ in range(n*m)]  # 36 x 189
    V = [[0]*rank for _ in range(m*p)]  # 36 x 189
    W = [[0]*rank for _ in range(p*n)]  # 36 x 189

    # Strassen 2x2 factors
    # U_s[r][i][j] : coefficient for input block A[i][j] in sub-multiplication r
    # V_s[r][j][k] : coefficient for input block B[j][k] in sub-multiplication r
    # W_s[r][i][k] : coefficient for output block C[i][k] in sub-multiplication r

    # Standard Strassen formulas for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = A22(B21-B11)
    # M3 = A11(B12-B22)
    # M4 = (A11+A12)(B22)
    # M5 = (A21+A22)(B11)
    # M6 = (A11-A21)(B12+B22)
    # M7 = (A12-A22)(B21+B11)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M6

    U_s = [
        # M1: A11 + A22
        [[1, 0], [0, 1]],
        # M2: A22
        [[0, 0], [0, 1]],
        # M3: A11
        [[1, 0], [0, 0]],
        # M4: A11 + A12
        [[1, 1], [0, 0]],
        # M5: A21 + A22
        [[0, 0], [1, 1]],
        # M6: A11 - A21
        [[1, 0], [-1, 0]],
        # M7: A12 - A22
        [[0, 1], [0, -1]]
    ]

    V_s = [
        # M1: B11 + B22
        [[1, 0], [0, 1]],
        # M2: B21 - B11
        [[-1, 0], [1, 0]],
        # M3: B12 - B22
        [[0, 1], [0, -1]],
        # M4: B22
        [[0, 0], [0, 1]],
        # M5: B11
        [[1, 0], [0, 0]],
        # M6: B12 + B22
        [[0, 1], [0, 1]],
        # M7: B21 + B11
        [[1, 0], [1, 0]]
    ]

    # W_s[r][i][k] in trace convention W[k*n+i, r], so for 2x2: W[k*2+i, r]
    # C11 = M1 + M4 - M5 - M7  -> (i=0,k=0)
    # C12 = M3 + M7             -> (i=1,k=0)
    # C21 = M2 + M5             -> (i=0,k=1)
    # C22 = M1 - M2 + M3 + M6  -> (i=1,k=1)
    # Index in flat: k*2 + i
    W_s = [
        # For each sub-multiplication r, W at positions (k*2+i):
        # r=0 (M1): C11+=1, C22+=1
        [1, 0, 0, 1],
        # r=1 (M2): C21+=1, C22+=-1
        [0, 1, 0, -1],
        # r=2 (M3): C12+=1, C22+=1
        [0, 0, 1, 1],
        # r=3 (M4): C11+=1
        [1, 0, 0, 0],
        # r=4 (M5): C11+=-1, C21+=1
        [-1, 0, 1, 0],
        # r=5 (M6): C22+=1
        [0, 0, 0, 1],
        # r=6 (M7): C11+=-1, C12+=1
        [-1, 1, 0, 0]
    ]

    col = 0
    for I in range(3):       # block row of A (and C)
        for J in range(3):   # block col of A / row of B
            for K in range(3):  # block col of B (and C)
                for r in range(7):
                    # Fill U: input A block (I, J), local 2x2
                    for u in range(2):
                        for v in range(2):
                            val = U_s[r][u][v]
                            if val != 0:
                                row_U = (I*2 + u) * 6 + (J*2 + v)
                                U[row_U][col] += val
                    # Fill V: input B block (J, K), local 2x2
                    for v in range(2):
                        for w in range(2):
                            val = V_s[r][v][w]
                            if val != 0:
                                row_V = (J*2 + v) * 6 + (K*2 + w)
                                V[row_V][col] += val
                    # Fill W: output C block (I, K), local 2x2
                    # W index in flat is (K*2+w)*6 + (I*2+u) = (output row K*2+w, col I*2+u)
                    # But the convention is W[k*n + i, r] where k=output row, i=output col
                    # In our 2x2 local: k=w (local output row), i=u (local output col)
                    # W_s[r] is indexed by k*2+i
                    for w in range(2):
                        for u in range(2):
                            val = W_s[r][w*2 + u]
                            if val != 0:
                                row_W = (K*2 + w) * 6 + (I*2 + u)
                                W[row_W][col] += val
                    col += 1

    return {"U": U, "V": V, "W": W}
