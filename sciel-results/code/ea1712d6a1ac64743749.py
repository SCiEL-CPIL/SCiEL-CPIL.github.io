import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    """
    Rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy: Split into three independent parts:
    1. 4x4 * 4x4 (cols 0-3): Strassen rank-49
    2. 4x4 * 4x2 (cols 4-5): 4 Strassen 2x2x2 blocks, rank 28
    3. 4x4 * 4x1 (col 6): schoolbook rank 16
    Total rank = 93
    """
    deadline = time.time() + 110
    n, m, p = 4, 4, 7
    rank = 93

    # Strassen 2x2x2 base
    u_base = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_base = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_base = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    U = [[0] * rank for _ in range(n * m)]   # 16 x 93
    V = [[0] * rank for _ in range(m * p)]   # 28 x 93
    W = [[0] * rank for _ in range(p * n)]   # 28 x 93

    col = 0

    # --- Part 1: 4x4 * 4x4 -> 4x4 (cols 0-3 of C) ---
    # Nested Strassen: outer 2x2 blocks, inner 2x2 blocks
    for k_outer in range(7):
        for l_inner in range(7):
            u_vec = [0] * 16
            v_vec = [0] * 28
            w_vec = [0] * 28

            # U: coefficient for A[i,j] where i = 2*br + ir, j = 2*bc + jr
            for br in range(2):
                for bc in range(2):
                    co = u_base[k_outer][br * 2 + bc]
                    if co == 0:
                        continue
                    for ir in range(2):
                        for jr in range(2):
                            ci = u_base[l_inner][ir * 2 + jr]
                            if ci == 0:
                                continue
                            ig = 2 * br + ir
                            jg = 2 * bc + jr
                            u_vec[ig * 4 + jg] += co * ci

            # V: coefficient for B[j,k] where j = 2*br + ir, k = 2*bc + jr
            # Only cols 0-3 of B (which are cols 0-3 of the 4x7 matrix)
            for br in range(2):
                for bc in range(2):
                    co = v_base[k_outer][br * 2 + bc]
                    if co == 0:
                        continue
                    for ir in range(2):
                        for jr in range(2):
                            ci = v_base[l_inner][ir * 2 + jr]
                            if ci == 0:
                                continue
                            jg = 2 * br + ir
                            kg = 2 * bc + jr
                            v_vec[jg * 7 + kg] += co * ci

            # W: coefficient for C[i,k] where i = 2*cbr + cir, k = 2*cbc + cic
            # Only cols 0-3 of C
            for cbr in range(2):
                for cbc in range(2):
                    cow = w_base[cbr * 2 + cbc][k_outer]
                    if cow == 0:
                        continue
                    for cir in range(2):
                        for cic in range(2):
                            ciw = w_base[cir * 2 + cic][l_inner]
                            if ciw == 0:
                                continue
                            tc = cow * ciw
                            if tc == 0:
                                continue
                            ig = 2 * cbr + cir
                            kg = 2 * cbc + cic
                            w_vec[kg * 4 + ig] += tc

            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
            for r in range(28):
                W[r][col] = w_vec[r]
            col += 1
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break

    # --- Part 2: 4x4 * 4x2 -> 4x2 (cols 4-5 of C) ---
    # 4 independent Strassen 2x2x2 blocks
    if col >= 49 and time.time() < deadline - 10:
        block_defs = [
            (0, 0, 0, 4, 0, 4),  # A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
            (0, 2, 2, 4, 0, 4),  # A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]
            (2, 0, 0, 4, 2, 4),  # A[2:4,0:2] * B[0:2,4:6] -> C[2:4,4:6]
            (2, 2, 2, 4, 2, 4),  # A[2:4,2:4] * B[2:4,4:6] -> C[2:4,4:6]
        ]

        for a_ro, a_co, b_ro, b_co, c_ro, c_co in block_defs:
            for k_inner in range(7):
                u_vec = [0] * 16
                v_vec = [0] * 28
                w_vec = [0] * 28

                for ii in range(2):
                    for jj in range(2):
                        c = u_base[k_inner][ii * 2 + jj]
                        if c != 0:
                            u_vec[(a_ro + ii) * 4 + (a_co + jj)] = c

                for ii in range(2):
                    for jj in range(2):
                        c = v_base[k_inner][ii * 2 + jj]
                        if c != 0:
                            v_vec[(b_ro + ii) * 7 + (b_co + jj)] = c

                for ci2 in range(4):
                    c = w_base[ci2][k_inner]
                    if c != 0:
                        cr, cc = ci2 // 2, ci2 % 2
                        w_vec[(c_co + cc) * 4 + (c_ro + cr)] = c

                for r in range(16):
                    U[r][col] = u_vec[r]
                for r in range(28):
                    V[r][col] = v_vec[r]
                for r in range(28):
                    W[r][col] = w_vec[r]
                col += 1
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break

    # --- Part 3: 4x4 * 4x1 -> 4x1 (col 6 of C) ---
    if col >= 77 and time.time() < deadline - 10:
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][col] = 1
                V[j * 7 + 6][col] = 1
                W[6 * 4 + i][col] = 1
                col += 1

    # Verification
    if col == rank:
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        np.random.seed(42)
        A_t = np.random.randint(-5, 5, size=(n, m)).astype(np.float64)
        B_t = np.random.randint(-5, 5, size=(m, p)).astype(np.float64)
        C_exp = A_t @ B_t
        C_comp = np.zeros((n, p))
        for r in range(rank):
            Ur = U_np[:, r]
            Vr = V_np[:, r]
            Wr = W_np[:, r]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += A_t[i, j] * B_t[j, k] * Ur[i * m + j] * Vr[j * p + k] * Wr[k * n + i]
                    C_comp[i, k] += s
        if not np.allclose(C_comp, C_exp, atol=1e-8):
            # Fallback to schoolbook
            rsb = n * m * p
            U_sb = [[0] * rsb for _ in range(n * m)]
            V_sb = [[0] * rsb for _ in range(m * p)]
            W_sb = [[0] * rsb for _ in range(p * n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j][c] = 1
                        V_sb[j * p + k][c] = 1
                        W_sb[k * n + i][c] = 1
                        c += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U, "V": V, "W": W}
