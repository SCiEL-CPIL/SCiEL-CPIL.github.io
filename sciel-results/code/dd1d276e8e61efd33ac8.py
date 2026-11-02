import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy:
    # Part 1: j=0, schoolbook 2x1x6 = 12 terms
    # Part 2: j=1..4, split into two 2x3x4 blocks, each with rank-17 decomposition
    #   2x3x4 = [2x2|2x1] * [2x2|2x2] * [1x2|1x2]
    #   = 2x2x2 (Strassen, 7) + 2x1x2 (4) + 2x2x1 (4) + 2x1x1 (2) = 17
    # Total rank = 12 + 2*17 = 46

    rank = 46
    U = [[0] * rank for _ in range(n * m)]   # (10, 46)
    V = [[0] * rank for _ in range(m * p)]   # (30, 46)
    W = [[0] * rank for _ in range(p * n)]   # (12, 46)

    col = 0

    # Part 1: j = 0, schoolbook 2x1x6
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col = 12

    # Part 2: j in 1..4, two blocks of 2x3x4
    for block_idx in range(2):
        j_base = 1 + block_idx * 2
        j1 = j_base      # first j in block
        j2 = j_base + 1  # second j in block
        j3 = j_base + 2  # third j in block

        # U indices: A[i, j] -> row i*m + j
        u0j1 = 0 * m + j1
        u0j2 = 0 * m + j2
        u0j3 = 0 * m + j3
        u1j1 = 1 * m + j1
        u1j2 = 1 * m + j2
        u1j3 = 1 * m + j3

        # V indices: B[j, k] -> row j*p + k
        v_j1k0 = j1 * p + 0
        v_j1k1 = j1 * p + 1
        v_j1k2 = j1 * p + 2
        v_j1k3 = j1 * p + 3
        v_j2k0 = j2 * p + 0
        v_j2k1 = j2 * p + 1
        v_j2k2 = j2 * p + 2
        v_j2k3 = j2 * p + 3
        v_j3k0 = j3 * p + 0
        v_j3k1 = j3 * p + 1
        v_j3k2 = j3 * p + 2
        v_j3k3 = j3 * p + 3

        # W indices: C[i, k] -> row k*n + i
        w00 = 0 * n + 0
        w01 = 1 * n + 0
        w02 = 2 * n + 0
        w03 = 3 * n + 0
        w10 = 0 * n + 1
        w11 = 1 * n + 1
        w12 = 2 * n + 1
        w13 = 3 * n + 1

        # === 2x2x2 Strassen for A_sub = [[A0j1, A0j2], [A1j1, A1j2]], B_sub = [[B_j1k0, B_j1k1], [B_j2k0, B_j2k1]], C_sub = [[C00, C01], [C10, C11]] ===
        # P1 = (a00+a11)(b00+b11)
        U[u0j1][col] += 1; U[u1j2][col] += 1
        V[v_j1k0][col] += 1; V[v_j2k1][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1

        # P2 = (a10+a11)b00
        U[u1j1][col] += 1; U[u1j2][col] += 1
        V[v_j1k0][col] += 1
        W[w10][col] += 1; W[w11][col] += -1
        col += 1

        # P3 = a00(b01-b11)
        U[u0j1][col] += 1
        V[v_j1k1][col] += 1; V[v_j2k1][col] += -1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1

        # P4 = a11(b10-b00)
        U[u1j2][col] += 1
        V[v_j2k0][col] += 1; V[v_j1k0][col] += -1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1

        # P5 = (a00+a01)b11
        U[u0j1][col] += 1; U[u0j2][col] += 1
        V[v_j2k1][col] += 1
        W[w00][col] += -1; W[w01][col] += 1
        col += 1

        # P6 = (a10-a00)(b00+b01)
        U[u1j1][col] += 1; U[u0j1][col] += -1
        V[v_j1k0][col] += 1; V[v_j1k1][col] += 1
        W[w11][col] += 1
        col += 1

        # P7 = (a01-a11)(b10+b11)
        U[u0j2][col] += 1; U[u1j2][col] += -1
        V[v_j2k0][col] += 1; V[v_j2k1][col] += 1
        W[w00][col] += 1
        col += 1

        # === 2x1x2 schoolbook: A0j3 * B_j3k0, A0j3 * B_j3k1 -> C02, C03 ===
        U[u0j3][col] = 1; V[v_j3k0][col] = 1; W[w02][col] = 1
        col += 1
        U[u0j3][col] = 1; V[v_j3k1][col] = 1; W[w03][col] = 1
        col += 1

        # === 2x2x1 schoolbook: A0j1*B_j1k2, A0j2*B_j2k2 -> C02; A1j1*B_j1k2, A1j2*B_j2k2 -> C12 ===
        U[u0j1][col] = 1; V[v_j1k2][col] = 1; W[w02][col] = 1
        col += 1
        U[u0j2][col] = 1; V[v_j2k2][col] = 1; W[w02][col] = 1
        col += 1
        U[u1j1][col] = 1; V[v_j1k2][col] = 1; W[w12][col] = 1
        col += 1
        U[u1j2][col] = 1; V[v_j2k2][col] = 1; W[w12][col] = 1
        col += 1

        # === 2x1x1: A0j3*B_j3k3 -> C03, A1j3*B_j3k3 -> C13 ===
        U[u0j3][col] = 1; V[v_j3k3][col] = 1; W[w03][col] = 1
        col += 1
        U[u1j3][col] = 1; V[v_j3k3][col] = 1; W[w13][col] = 1
        col += 1

    # Verify
    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat, dtype=np.int64)
        V_np = np.array(V_mat, dtype=np.int64)
        W_np = np.array(W_mat, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if result != 1:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback: rank 60 trivial
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
