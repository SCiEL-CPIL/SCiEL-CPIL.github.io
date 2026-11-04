import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# ---- 2x2x2 Strassen coefficient tables (blocks indexed 0=A00,1=A01,2=A10,3=A11) ----
ALPHA = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # s0: A00 + A11
    {0: 0, 1: 0, 2: 1, 3: 1},   # s1: A10 + A11
    {0: 1, 1: 0, 2: 0, 3: 0},   # s2: A00
    {0: 0, 1: 0, 2: 0, 3: 1},   # s3: A11
    {0: 1, 1: 1, 2: 0, 3: 0},   # s4: A00 + A01
    {0: -1, 1: 0, 2: 1, 3: 0},  # s5: A10 - A00
    {0: 0, 1: 1, 2: 0, 3: -1},  # s6: A01 - A11
]
BETA = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # s0: B00 + B11
    {0: 1, 1: 0, 2: 0, 3: 0},   # s1: B00
    {0: 0, 1: 1, 2: 0, 3: -1},  # s2: B01 - B11
    {0: -1, 1: 0, 2: 1, 3: 0},  # s3: B10 - B00
    {0: 0, 1: 0, 2: 0, 3: 1},   # s4: B11
    {0: 1, 1: 1, 2: 0, 3: 0},   # s5: B00 + B01
    {0: 0, 1: 0, 2: 1, 3: 1},   # s6: B10 + B11
]
RECON = [
    [(0, 0, 1), (1, 1, 1)],     # s0
    [(1, 0, 1), (1, 1, -1)],    # s1
    [(0, 1, 1), (1, 1, 1)],     # s2
    [(0, 0, 1), (1, 0, 1)],     # s3
    [(0, 0, -1), (0, 1, 1)],    # s4
    [(1, 1, 1)],                # s5
    [(0, 0, 1)],                # s6
]


def _build():
    n, m, p = 4, 4, 6
    # Part 1: 4x4 * 4x4  (Kronecker of two 2x2 Strassen)  -> 7*7 = 49 terms
    # Part 2: 4x4 * 4x2  (2x2 Strassen per 2x2 block)     -> 4*7 = 28 terms
    R = 77
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # ---------- Part 1: 4x4 * 4x4 ----------
    # outer (t): 4x4->2x2 Strassen ; inner (s): 2x2->2x2 Strassen
    col = 0
    for t in range(7):
        for s in range(7):
            # U: A[(2*I+J),(2*i0+j0)] where I,J from outer block, i0,j0 inner
            for I in range(2):
                for J in range(2):
                    a_idx = 2 * I + J  # block index in outer
                    if ALPHA[t][a_idx] == 0:
                        continue
                    for i0 in range(2):
                        for j0 in range(2):
                            b_idx = 2 * i0 + j0
                            if ALPHA[s][b_idx] == 0:
                                continue
                            val = ALPHA[t][a_idx] * ALPHA[s][b_idx]
                            if val != 0:
                                U[(2 * I + i0) * 4 + (2 * J + j0), col] += val
            # V
            for J in range(2):
                for K in range(2):
                    b_idx = 2 * J + K
                    if BETA[t][b_idx] == 0:
                        continue
                    for j0 in range(2):
                        for k0 in range(2):
                            c_idx = 2 * j0 + k0
                            if BETA[s][c_idx] == 0:
                                continue
                            val = BETA[t][b_idx] * BETA[s][c_idx]
                            if val != 0:
                                V[(2 * J + j0) * 4 + (2 * K + k0), col] += val
            # W: C[(2*K+k0),(2*I+i0)]
            for K in range(2):
                for I in range(2):
                    # outer recon (I,K)
                    outer_g = 0
                    for (ra, rb, ga) in RECON[t]:
                        if ra == I and rb == K:
                            outer_g += ga
                    if outer_g == 0:
                        continue
                    for i0 in range(2):
                        for k0 in range(2):
                            inner_g = 0
                            for (rc, rd, gc) in RECON[s]:
                                if rc == k0 and rd == i0:
                                    inner_g += gc
                            if inner_g == 0:
                                continue
                            val = outer_g * inner_g
                            if val != 0:
                                W[(2 * K + k0) * 4 + (2 * I + i0), col] += val
            col += 1

    # ---------- Part 2: 4x4 * 4x2 ----------
    # 4x4 * 4x2 split into 2x2 * 2x2 blocks at A-rows {0,1},{2,3} and B-cols {0,1},{2,3}
    # 2x2 Strassen per block -> 4 blocks * 7 = 28 terms
    blocks = [
        (0, 0),  # A rows 0-1, B cols 0-1
        (0, 2),  # A rows 0-1, B cols 2-3
        (2, 0),  # A rows 2-3, B cols 0-1
        (2, 2),  # A rows 2-3, B cols 2-3
    ]
    for (ar, bc) in blocks:
        for s in range(7):
            # U: A[(ar+i),(ar+j)]
            for i in range(2):
                for j in range(2):
                    b_idx = 2 * i + j
                    if ALPHA[s][b_idx] != 0:
                        U[(ar + i) * 4 + (ar + j), col] += ALPHA[s][b_idx]
            # V: B[(bc+j),(bc+k)]  (B is 4x4 here, cols bc,bc+1)
            for j in range(2):
                for k in range(2):
                    c_idx = 2 * j + k
                    if BETA[s][c_idx] != 0:
                        V[(bc + j) * 6 + (bc + k), col] += BETA[s][c_idx]
            # W: C[(bc+k),(ar+i)]
            for k in range(2):
                for i in range(2):
                    g = 0
                    for (rc, rd, gc) in RECON[s]:
                        if rc == k and rd == i:
                            g += gc
                    if g != 0:
                        W[(bc + k) * 4 + (ar + i), col] += g
            col += 1

    assert col == R
    return U, V, W


def _verify(U, V, W, n=4, m=4, p=6, tol=1e-9):
    # tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]  should equal 1
    R = U.shape[1]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _to_list(mat):
    return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat]


def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    n, m, p = 4, 4, 6

    U, V, W = _build()
    R = U.shape[1]

    # Build contribution matrix M[(i,j,k), r] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    num_tensors = n * m * p
    M = np.zeros((num_tensors, R))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(R):
                    M[idx, r] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                idx += 1

    # Greedy iterative rank reduction: repeatedly remove any column whose
    # contribution vector lies in the span of the remaining columns.
    U_c, V_c, W_c = U, V, W
    R_c = R
    for _ in range(R_c):
        if time.time() > deadline - 25.0:
            break
        removed = False
        for r_remove in range(R_c):
            if time.time() > deadline - 25.0:
                break
            mask = np.ones(R_c, dtype=bool)
            mask[r_remove] = False
            M_rem = M_c = M[:, mask]
            target = M[:, r_remove]
            try:
                coeffs, res, rank_rem, _ = np.linalg.lstsq(M_rem, target, rcond=1e-10)
                recon = M_rem @ coeffs
                err = np.linalg.norm(target - recon)
                if err < 1e-6 * max(1.0, np.linalg.norm(target)):
                    # candidate: remove r_remove
                    new_U = U_c[:, mask]
                    new_V = V_c[:, mask]
                    new_W = W_c[:, mask]
                    if _verify(new_U, new_V, new_W, n, m, p, 1e-8):
                        U_c, V_c, W_c = new_U, new_V, new_W
                        R_c -= 1
                        M = M[:, mask]
                        removed = True
                        break
            except Exception:
                pass
        if not removed:
            break

    return {"U": _to_list(U_c), "V": _to_list(V_c), "W": _to_list(W_c)}
