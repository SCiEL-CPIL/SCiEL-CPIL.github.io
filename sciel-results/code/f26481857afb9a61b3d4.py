import numpy as np

def construct():
    """
    Rank-98 exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Built from two independent rank-49 Strassen decompositions of 4x4x4.
    """
    n, m, p = 4, 4, 8

    # ---------- Build rank-49 decomposition for 4x4x4 ----------
    # Outer Strassen (block level): 7 block products M_t = A^(t) * B^(t)
    # Each M_t is 2x2 of 2x2 blocks, so we need inner Strassen (7 scalar mults)
    # Total: 7*7 = 49 rank terms.

    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # alpha[t]: coefficient of A-block P in A^(t)
    alpha = {
        1: {0: 1, 3: 1},
        2: {2: 1, 3: 1},
        3: {0: 1},
        4: {3: 1},
        5: {0: 1, 1: 1},
        6: {2: 1, 0: -1},
        7: {1: 1, 3: -1},
    }

    # beta[t]: coefficient of B-block Q in B^(t)
    beta = {
        1: {0: 1, 3: 1},
        2: {0: 1},
        3: {1: 1, 3: -1},
        4: {2: 1, 0: -1},
        5: {3: 1},
        6: {0: 1, 1: 1},
        7: {2: 1, 3: 1},
    }

    # kappa[t]: coefficient of M_t in C-block PQ
    kappa = {
        1: {0: 1, 3: 1},
        2: {2: 1, 3: -1},
        3: {1: 1, 3: 1},
        4: {0: 1, 2: 1},
        5: {0: -1, 1: 1},
        6: {3: 1},
        7: {0: 1},
    }

    # Inner Strassen for 2x2 matrices:
    # u_s = sum_ab du[s][ab] * M_ab
    # v_s = sum_ab dv[s][ab] * N_ab
    # M*N entry (a,b) = sum_s gamma[s][(a,b)] * (u_s * v_s)

    du = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): 1},
        3: {(0,0): 1},
        4: {(1,1): 1},
        5: {(0,0): 1, (0,1): 1},
        6: {(1,0): 1, (0,0): -1},
        7: {(0,1): 1, (1,1): -1},
    }

    dv = {
        1: {(0,0): 1, (1,1): 1},
        2: {(0,0): 1},
        3: {(0,1): 1, (1,1): -1},
        4: {(1,0): 1, (0,0): -1},
        5: {(1,1): 1},
        6: {(0,0): 1, (0,1): 1},
        7: {(1,0): 1, (1,1): 1},
    }

    gamma = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): -1},
        3: {(0,1): 1, (1,1): 1},
        4: {(0,0): 1, (1,0): 1},
        5: {(0,1): 1, (0,0): -1},
        6: {(1,1): 1},
        7: {(0,0): 1},
    }

    # Build U444 (16 x 49), V444 (16 x 49), W444 (16 x 49)
    # U index: i*4 + j  (row i, col j of A)
    # V index: j*4 + k  (row j, col k of B)
    # W index: k*4 + i  (row k, col i of C) -- trace convention W[k*n+i, r]

    U444 = np.zeros((16, 49), dtype=int)
    V444 = np.zeros((16, 49), dtype=int)
    W444 = np.zeros((16, 49), dtype=int)

    for t in range(1, 8):
        for s in range(1, 8):
            r = (t - 1) * 7 + (s - 1)

            # U contribution: u_s = sum_ab du[s][ab] * A^(t)_ab
            # A^(t)_ab = sum_P alpha[t][P] * A_{2*Prow+a, 2*Pcol+b}
            for a in range(2):
                for b in range(2):
                    cu = du[s].get((a, b), 0)
                    if cu == 0:
                        continue
                    for P, av in alpha[t].items():
                        c = cu * av
                        if c == 0:
                            continue
                        i = 2 * (P // 2) + a
                        j = 2 * (P % 2) + b
                        U444[i * 4 + j, r] += c

            # V contribution: v_s = sum_ab dv[s][ab] * B^(t)_ab
            # B^(t)_ab = sum_Q beta[t][Q] * B_{2*Qrow+a, 2*Qcol+b}
            for a in range(2):
                for b in range(2):
                    cv = dv[s].get((a, b), 0)
                    if cv == 0:
                        continue
                    for Q, bv in beta[t].items():
                        c = cv * bv
                        if c == 0:
                            continue
                        j = 2 * (Q // 2) + a
                        k = 2 * (Q % 2) + b
                        V444[j * 4 + k, r] += c

            # W contribution: C_PQ[ab] gets kappa[t][PQ] * gamma[s][(a,b)] * m_s
            # C_PQ[ab] is at global row i=2*PQrow+a, col k=2*PQcol+b
            # W index = k*n + i = k*4 + i
            for a in range(2):
                for b in range(2):
                    cm = gamma[s].get((a, b), 0)
                    if cm == 0:
                        continue
                    for PQ, kv in kappa[t].items():
                        c = cm * kv
                        if c == 0:
                            continue
                        i = 2 * (PQ // 2) + a
                        k = 2 * (PQ % 2) + b
                        W444[k * 4 + i, r] += c

    # Verify 4x4x4 decomposition with random test
    for trial in range(3):
        A_test = np.random.randint(-3, 4, size=(4, 4)).astype(float)
        B_test = np.random.randint(-3, 4, size=(4, 4)).astype(float)
        C_expected = A_test @ B_test
        C_computed = np.zeros((4, 4), dtype=float)
        for r in range(49):
            u_val = float(U444[:, r] @ A_test.flatten())
            v_val = float(V444[:, r] @ B_test.flatten())
            prod = u_val * v_val
            C_computed += (W444[:, r].reshape(4, 4).T) * prod
        # W444[k*4+i, r] means entry (k,i) gets W444[k*4+i, r] * prod
        # C[k,i] = sum_r W444[k*4+i, r] * u_r * v_r
        # So C_computed[k,i] = sum_r W444[k*4+i, r] * (U444[:,r] @ A) * (V444[:,r] @ B)
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Try alternate interpretation
            C_computed2 = np.zeros((4, 4), dtype=float)
            for r in range(49):
                u_val = float(U444[:, r] @ A_test.flatten())
                v_val = float(V444[:, r] @ B_test.flatten())
                prod = u_val * v_val
                for idx in range(16):
                    w_val = W444[idx, r]
                    if w_val != 0:
                        k = idx // 4
                        i = idx % 4
                        C_computed2[k, i] += w_val * prod
            if not np.allclose(C_computed2, C_expected, atol=1e-9):
                raise ValueError("4x4x4 decomposition verification failed")

    # ---------- Expand to 4x4x8 (rank 98) ----------
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)

    # Copy 1: k in [0, 4)
    U[:, 0:49] = U444
    for j in range(4):
        for k in range(4):
            V[8 * j + k, 0:49] = V444[4 * j + k, :]
    for k in range(4):
        for i in range(4):
            W[4 * k + i, 0:49] = W444[4 * k + i, :]

    # Copy 2: k in [4, 8)
    U[:, 49:98] = U444
    for j in range(4):
        for k in range(4, 8):
            V[8 * j + k, 49:98] = V444[4 * j + (k - 4), :]
    for k in range(4, 8):
        for i in range(4):
            W[4 * k + i, 49:98] = W444[4 * (k - 4) + i, :]

    # Final verification for 4x4x8
    A_test = np.random.randint(-3, 4, size=(4, 4)).astype(float)
    B_test = np.random.randint(-3, 4, size=(4, 8)).astype(float)
    C_expected = A_test @ B_test
    C_computed = np.zeros((4, 8), dtype=float)
    for r in range(rank):
        u_val = float(U[:, r] @ A_test.flatten())
        v_val = float(V[:, r] @ B_test.flatten())
        prod = u_val * v_val
        for idx in range(32):
            w_val = W[idx, r]
            if w_val != 0:
                k = idx // 4
                i = idx % 4
                C_computed[k, i] += w_val * prod
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        raise ValueError("4x4x8 decomposition verification failed")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
