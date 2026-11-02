import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 4, 4, 6

    # ---- Winograd rank-7 decomposition for 2x2*2x2 ----
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = [[0.0] * 7 for _ in range(4)]
    V2 = [[0.0] * 7 for _ in range(4)]
    W2 = [[0.0] * 7 for _ in range(4)]
    # term 0: (a00+a11)*(b00+b11) -> c00, c11
    U2[0][0] = 1.0; U2[3][0] = 1.0
    V2[0][0] = 1.0; V2[3][0] = 1.0
    W2[0][0] = 1.0; W2[3][0] = 1.0
    # term 1: (a10+a11)*b00 -> c10
    U2[2][1] = 1.0; U2[3][1] = 1.0
    V2[0][1] = 1.0
    W2[2][1] = 1.0
    # term 2: a00*(b01-b11) -> c01
    U2[0][2] = 1.0
    V2[1][2] = 1.0; V2[3][2] = -1.0
    W2[1][2] = 1.0
    # term 3: a11*(b10-b00) -> c10
    U2[3][3] = 1.0
    V2[2][3] = 1.0; V2[0][3] = -1.0
    W2[2][3] = 1.0
    # term 4: (a00+a01)*b11 -> c00(-), c01
    U2[0][4] = 1.0; U2[1][4] = 1.0
    V2[3][4] = 1.0
    W2[0][4] = -1.0; W2[1][4] = 1.0
    # term 5: (a10-a00)*(b00+b01) -> c11
    U2[2][5] = 1.0; U2[0][5] = -1.0
    V2[0][5] = 1.0; V2[1][5] = 1.0
    W2[3][5] = 1.0
    # term 6: (a01-a11)*(b10+b11) -> c00, c11
    U2[1][6] = 1.0; U2[3][6] = -1.0
    V2[2][6] = 1.0; V2[3][6] = 1.0
    W2[0][6] = 1.0; W2[3][6] = 1.0

    def verify_2x2(U2, V2, W2):
        Ua = np.array(U2, dtype=float); Va = np.array(V2, dtype=float); Wa = np.array(W2, dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = np.dot(Ua[i * 2 + j], Va[j * 2 + k] * Wa[k * 2 + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    winograd_ok = verify_2x2(U2, V2, W2)

    # ---- Recursive Strassen 4x4*4x4 (rank 49) using Winograd 2x2*2x2 ----
    def strassen_4x4_4x4():
        # 7 Strassen terms, each a 2x2*2x2 product (rank 7) -> total 49
        U = [[0.0] * 49 for _ in range(16)]
        V = [[0.0] * 49 for _ in range(16)]
        W = [[0.0] * 49 for _ in range(16)]

        def Uidx(i, j): return i * 4 + j
        def Vidx(j, k): return j * 4 + k
        def Widx(k, i): return k * 4 + i

        # Strassen terms: (M, A-block, B-block, sign)
        # A-block signs per term (for the two A sub-blocks involved)
        A_signs = [
            [(0, 0, 1), (1, 1, 1)],   # M1
            [(1, 0, 1), (1, 1, 1)],   # M2
            [(0, 0, 1)],              # M3
            [(1, 1, 1)],              # M4
            [(0, 0, 1), (0, 1, 1)],   # M5
            [(1, 0, 1), (0, 0, -1)],  # M6
            [(0, 1, 1), (1, 1, -1)],  # M7
        ]
        # B-block signs per term
        B_signs = [
            [(0, 0, 1), (1, 1, 1)],   # M1
            [(0, 0, 1)],              # M2
            [(0, 1, 1), (1, 1, -1)],  # M3
            [(1, 0, 1), (0, 0, -1)],  # M4
            [(1, 1, 1)],              # M5
            [(0, 0, 1), (0, 1, 1)],   # M6
            [(1, 0, 1), (1, 1, 1)],   # M7
        ]
        # C-block contributions: (C-block, sign) per term
        C_final_terms = [
            [0, 3, 4, 6],   # C00: M1,M4,M5,M7
            [2, 4],         # C01: M3,M5
            [1, 3],         # C10: M2,M4
            [0, 1, 2, 5],   # C11: M1,M2,M3,M6
        ]
        C_final_signs = [
            [1, 1, -1, 1],  # C00
            [1, 1],         # C01
            [1, 1],         # C10
            [1, -1, 1, 1],  # C11
        ]

        for t in range(7):
            base = t * 7
            # U: for each A sub-block (bi,bj) with sign sa
            for (bi, bj, sa) in A_signs[t]:
                for di in range(2):
                    for dj in range(2):
                        U[Uidx(bi * 2 + di, bj * 2 + dj)][base + U2[di * 2 + dj]] += sa * 1.0
            # V: for each B sub-block (bj,bk) with sign sb
            for (bj, bk, sb) in B_signs[t]:
                for dj in range(2):
                    for dk in range(2):
                        V[Vidx(bj * 2 + dj, bk * 2 + dk)][base + V2[dj * 2 + dk]] += sb * 1.0
            # W: contributions to C blocks
            for ci, cblock in enumerate(range(4)):
                for (term_idx, sign) in zip(C_final_terms[ci], C_final_signs[ci]):
                    if term_idx == t:
                        (ci_i, ci_j) = (ci // 2, ci % 2)
                        for dk in range(2):
                            for di in range(2):
                                W[Widx(ci_j * 2 + dk, ci_i * 2 + di)][base + W2[dk * 2 + di]] += sign * 1.0
        return U, V, W

    def verify_4x4(U, V, W, n, m, p):
        Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # ---- 4x4*2x2 (rank 28) using 4 Winograd blocks ----
    def fourx4_2x2():
        # A is 4x4, B is 4x2, C is 4x2
        # Split A into 2x2 blocks A00,A01,A10,A11; B into B0 (rows 0-1, 2x2), B1 (rows 2-3, 2x2)
        # C0 (rows 0-1, 2x2) = A00*B0 + A01*B1
        # C1 (rows 2-3, 2x2) = A10*B0 + A11*B1
        # 4 Winograd 2x2*2x2 products:
        # P0 = A00*B0, P1 = A01*B1, P2 = A10*B0, P3 = A11*B1
        # C0 = P0 + P1, C1 = P2 + P3
        U = [[0.0] * 28 for _ in range(16)]
        V = [[0.0] * 28 for _ in range(8)]
        W = [[0.0] * 28 for _ in range(8)]
        def Uidx(i, j): return i * 4 + j
        def Vidx(j, k): return j * 2 + k
        def Widx(k, i): return k * 4 + i
        # products: (A-block, B-block, C-block)
        prods = [
            ((0, 0), (0, 0), (0, 0)),  # P0 = A00*B0 -> C0
            ((0, 1), (1, 0), (0, 0)),  # P1 = A01*B1 -> C0
            ((1, 0), (0, 0), (1, 0)),  # P2 = A10*B0 -> C1
            ((1, 1), (1, 0), (1, 0)),  # P3 = A11*B1 -> C1
        ]
        for t in range(4):
            base = t * 7
            (ai, aj), (bi, bk), (ci, cj) = prods[t]
            for di in range(2):
                for dj in range(2):
                    U[Uidx(ai * 2 + di, aj * 2 + dj)][base + U2[di * 2 + dj]] = 1.0
            for dj in range(2):
                for dk in range(2):
                    V[Vidx(bi * 2 + dj, dk)][base + V2[dj * 2 + dk]] = 1.0
            for dk in range(2):
                for di in range(2):
                    W[Widx(dk, ci * 2 + di)][base + W2[dk * 2 + di]] = 1.0
        return U, V, W

    # ---- Build the full 4x4*4x6 decomposition (rank 77) ----
    if winograd_ok and time.time() < deadline - 20:
        U44, V44, W44 = strassen_4x4_4x4()
        U42, V42, W42 = fourx4_2x2()

        # Verify 4x4*4x4 piece
        ok44 = verify_4x4(U44, V44, W44, 4, 4, 4)
        # Verify 4x4*2x2 piece
        ok42 = verify_4x4(U42, V42, W42, 4, 2, 2)

        if ok44 and ok42:
            # Combine: B columns 0-3 use 4x4 piece, columns 4-5 use 2x2 piece
            rank = 49 + 28
            U = [[0.0] * rank for _ in range(16)]
            V = [[0.0] * rank for _ in range(24)]  # m*p = 4*6
            W = [[0.0] * rank for _ in range(24)]  # p*n = 6*4

            # First 49 columns: 4x4*4x4 for B columns 0-3
            for r in range(49):
                for i in range(16):
                    U[i][r] = U44[i][r]
                for j in range(4):
                    for k in range(4):
                        V[j * 6 + k][r] = V44[j * 4 + k][r]
                        W[k * 4 + i // 4][r] = W44[k * 4 + i % 4][r]
            # Next 28 columns: 4x4*2x2 for B columns 4-5
            for r in range(28):
                rr = 49 + r
                for i in range(16):
                    U[i][rr] = U42[i][r]
                for j in range(4):
                    for k in range(2):
                        V[j * 6 + (4 + k)][rr] = V42[j * 2 + k][r]
                        W[(4 + k) * 4 + i // 4][rr] = W42[k * 4 + i % 4][r]

            # Full verification
            if verify_4x4(U, V, W, 4, 4, 6):
                return {"U": U, "V": V, "W": W}

    # ---- Fallback: schoolbook rank 96 ----
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
