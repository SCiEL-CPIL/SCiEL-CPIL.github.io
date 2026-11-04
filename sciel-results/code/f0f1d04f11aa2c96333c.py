import time

def construct():
    start = time.time()
    DEADLINE = 150.0
    MARGIN = 3.0

    def time_left():
        return DEADLINE - MARGIN - (time.time() - start)

    n, m, p = 4, 4, 7

    # ---- Trivial exact decomposition (guaranteed valid, rank = n*m*p) ----
    def trivial():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][t] = 1
                    V[j * p + k][t] = 1
                    W[k * n + i][t] = 1
                    t += 1
        return {"U": U, "V": V, "W": W}

    # ---- Strassen base for 2x2 x 2x2 -> 2x2 (rank 7) ----
    def strassen222():
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = 1; W[3][4] = -1
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = -1
        return U, V, W

    # ---- Verification: reconstruct multiplication tensor and compare ----
    def verify(U, V, W, rank):
        # tensor[i, j, k] = A[i,j] * B[j,k]  -> index (i*m+j) * (m*p) * ... 
        # We compute C[i,k] = sum_j A[i,j] B[j,k]; the tensor element is C[i,k] per (i,j,k) pair.
        # Decomposition: sum_t U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] must equal A[i,j]*B[j,k].
        # Check via random-ish integer sample of all (i,j,k) is 4*4*7=112, do all.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # target = 1 (A[i,j]=1, B[j,k]=1) in the "symbolic" basis? 
                    # Actually the tensor T[i*m+j, j*p+k, k*n+i] = 1 for the multiplication tensor.
                    # The decomposition must satisfy sum_t U[..][t]*V[..][t]*W[..][t] = 1 for each (i,j,k).
                    s = 0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if s != 1:
                        return False
        return True

    # ---- Block decomposition: Strassen(4x4x4) + schoolbook(4x4x3) ----
    def block_decomp():
        sU, sV, sW = strassen222()
        total_rank = 49 + 48  # 7*7 + 4*4*3
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        def a_idx(br, bc, lr, lc, mm):
            return (br * 2 + lr) * mm + (bc * 2 + lc)

        def b_idx(br, bc, lr, lc, pp):
            return (br * 2 + lr) * pp + (bc * 2 + lc)

        def c_idx(br, bc, lr, lc, nn):
            # W index = k*n + i ; k = col = bc*2+lc ; i = row = br*2+lr
            return (bc * 2 + lc) * nn + (br * 2 + lr)

        steps = [
            ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
            ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
            ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}),
            ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}),
            ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
            ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}),
            ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}),
        ]

        for step, (a_blocks, b_blocks) in enumerate(steps):
            for base_term in range(7):
                col = step * 7 + base_term
                # U
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        cu = sU[local_idx][base_term]
                        if cu == 0:
                            continue
                        for (br, bc), ca in a_blocks.items():
                            if ca == 0:
                                continue
                            U[a_idx(br, bc, lr, lc, m)][col] += cu * ca
                # V
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        cv = sV[local_idx][base_term]
                        if cv == 0:
                            continue
                        for (br, bc), cb in b_blocks.items():
                            if cb == 0:
                                continue
                            V[b_idx(br, bc, lr, lc, 4)][col] += cv * cb
                # W
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        cw = sW[local_idx][base_term]
                        if cw == 0:
                            continue
                        W[c_idx(br=0, bc=0, lr=lr, lc=lc, nn=n)]  # placeholder no-op
                        # correct: use the C-block mapping (br, bc) = (0,0) for the 4x4 part
                        W[c_idx(0, 0, lr, lc, n)][col] += cw

        # ---- Schoolbook for columns 4..6 (4x4 x 4x3) ----
        base = 49
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(4, 7):
                    U[i * m + j][base + t] = 1
                    V[j * p + k][base + t] = 1
                    W[k * n + i][base + t] = 1
                    t += 1

        return {"U": U, "V": V, "W": W}, total_rank

    # ---- Try the structural decomposition, verify, else fall back ----
    try:
        if time_left() > 2.0:
            cand, rank = block_decomp()
            if verify(cand["U"], cand["V"], cand["W"], rank):
                return cand
    except Exception:
        pass

    # Safety fallback: guaranteed valid trivial decomposition
    return trivial()
