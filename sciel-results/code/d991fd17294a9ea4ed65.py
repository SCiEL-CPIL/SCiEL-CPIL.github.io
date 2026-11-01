import time

def construct():
    start_time = time.time()
    n = 4
    m = 4
    p = 4
    R = 49

    U = [[0]*R for _ in range(16)]
    V = [[0]*R for _ in range(16)]
    W = [[0]*R for _ in range(16)]

    def a_idx(i, j):
        return i * 4 + j

    def b_idx(i, j):
        return i * 4 + j

    def w_idx(i, k):
        return k * 4 + i

    # Inner 2x2 Strassen with proper 4-entry indexing
    # Entry index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]

    # t=0: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1

    # t=1: (A10+A11)B00 -> C10+=1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1

    # t=2: A00(B01-B11) -> C01+=1
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1

    # t=3: A11(B10-B00) -> C00+=1, C10+=1
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1

    # t=4: (A00+A01)B11 -> C00-=1, C01+=1
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1

    # t=5: (A10-A00)(B00+B01) -> C11+=1
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1

    # t=6: (A01-A11)(B10+B11) -> C00+=1
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # Outer Strassen: 7 steps on 2x2 blocks
    A_terms = [
        [(0,1), (3,1)],
        [(2,1), (3,1)],
        [(0,1)],
        [(3,1)],
        [(0,1), (1,1)],
        [(2,1), (0,-1)],
        [(1,1), (3,-1)],
    ]
    B_terms = [
        [(0,1), (3,1)],
        [(0,1)],
        [(1,1), (3,-1)],
        [(2,1), (0,-1)],
        [(3,1)],
        [(0,1), (1,1)],
        [(2,1), (3,1)],
    ]
    C_contrib = [
        [(0,1), (3,1)],
        [(2,1)],
        [(1,1)],
        [(0,1), (2,1)],
        [(0,-1), (1,1)],
        [(3,1)],
        [(0,1)],
    ]

    outer_A = [[0]*4 for _ in range(7)]
    outer_B = [[0]*4 for _ in range(7)]
    outer_C = [[0]*4 for _ in range(7)]

    for s in range(7):
        for (block, coeff) in A_terms[s]:
            outer_A[s][block] += coeff
        for (block, coeff) in B_terms[s]:
            outer_B[s][block] += coeff
        for (block, coeff) in C_contrib[s]:
            outer_C[s][block] += coeff

    for s in range(7):
        for t in range(7):
            col = s * 7 + t

            for idx in range(4):
                coeff_inner = U2[idx][t]
                if coeff_inner != 0:
                    bu = idx // 2
                    bv = idx % 2
                    coeff_outer = outer_A[s][idx]
                    if coeff_outer != 0:
                        total = coeff_inner * coeff_outer
                        for i in range(2):
                            for j in range(2):
                                gi = bu * 2 + i
                                gj = bv * 2 + j
                                U[a_idx(gi, gj)][col] += total

            for idx in range(4):
                coeff_inner = V2[idx][t]
                if coeff_inner != 0:
                    bu = idx // 2
                    bv = idx % 2
                    coeff_outer = outer_B[s][idx]
                    if coeff_outer != 0:
                        total = coeff_inner * coeff_outer
                        for i in range(2):
                            for j in range(2):
                                gi = bu * 2 + i
                                gj = bv * 2 + j
                                V[b_idx(gi, gj)][col] += total

            for idx in range(4):
                coeff_inner = W2[idx][t]
                if coeff_inner != 0:
                    i_local = idx % 2
                    k_local = idx // 2
                    bu = i_local
                    bv = k_local
                    block_idx = bu * 2 + bv
                    coeff_outer = outer_C[s][block_idx]
                    if coeff_outer != 0:
                        total = coeff_inner * coeff_outer
                        for i_sub in range(2):
                            for k_sub in range(2):
                                gi = bu * 2 + i_sub
                                gk = bv * 2 + k_sub
                                W[w_idx(gi, gk)][col] += total

    # Verify the decomposition
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0
                for r in range(R):
                    val += U[a_idx(i, j)][r] * V[b_idx(j, k)][r] * W[w_idx(i, k)][r]
                if val != 1:
                    # Fallback: return schoolbook rank 64
                    U_fb = [[0]*64 for _ in range(16)]
                    V_fb = [[0]*64 for _ in range(16)]
                    W_fb = [[0]*64 for _ in range(16)]
                    for i2 in range(4):
                        for j2 in range(4):
                            for k2 in range(4):
                                r = i2 * 16 + j2 * 4 + k2
                                U_fb[a_idx(i2, j2)][r] = 1
                                V_fb[b_idx(j2, k2)][r] = 1
                                W_fb[w_idx(i2, k2)][r] = 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}

    elapsed = time.time() - start_time
    if elapsed > 100:  # Safety margin
        U_fb = [[0]*64 for _ in range(16)]
        V_fb = [[0]*64 for _ in range(16)]
        W_fb = [[0]*64 for _ in range(16)]
        for i2 in range(4):
            for j2 in range(4):
                for k2 in range(4):
                    r = i2 * 16 + j2 * 4 + k2
                    U_fb[a_idx(i2, j2)][r] = 1
                    V_fb[b_idx(j2, k2)][r] = 1
                    W_fb[w_idx(i2, k2)][r] = 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
