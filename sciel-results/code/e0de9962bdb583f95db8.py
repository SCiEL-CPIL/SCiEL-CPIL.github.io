import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 108.0

    n, m, p = 6, 6, 6
    nm, mp, pn = 36, 36, 36

    # Build the rank-216 schoolbook decomposition as a baseline
    rank_school = 216
    U_s = np.zeros((nm, rank_school))
    V_s = np.zeros((mp, rank_school))
    W_s = np.zeros((pn, rank_school))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_s[i * m + j, col] = 1.0
                V_s[j * p + k, col] = 1.0
                W_s[k * n + i, col] = 1.0

    # Build a rank-189 Strassen (2x2 blocks of 3x3) decomposition
    # 3x3 schoolbook kernel
    n3 = 3
    r3 = 27
    U3 = np.zeros((9, r3))
    V3 = np.zeros((9, r3))
    W3 = np.zeros((9, r3))
    for i3 in range(n3):
        for j3 in range(n3):
            for k3 in range(n3):
                col3 = i3 * 9 + j3 * 3 + k3
                U3[i3 * 3 + j3, col3] = 1.0
                V3[j3 * 3 + k3, col3] = 1.0
                W3[k3 * 3 + i3, col3] = 1.0

    total_rank = 7 * r3  # 189
    U6 = np.zeros((nm, total_rank))
    V6 = np.zeros((mp, total_rank))
    W6 = np.zeros((pn, total_rank))

    def add_term(start, a_blks, b_blks, c_blks):
        for r in range(r3):
            um = U3[:, r].reshape(3, 3)
            vm = V3[:, r].reshape(3, 3)
            wm = W3[:, r].reshape(3, 3)
            for bi, cf in a_blks:
                br, bc = bi // 2, bi % 2
                for ii in range(3):
                    for jj in range(3):
                        v = cf * um[ii, jj]
                        if v != 0:
                            U6[(br*3+ii)*6 + (bc*3+jj), start+r] += v
            for bi, cf in b_blks:
                br, bc = bi // 2, bi % 2
                for jj in range(3):
                    for kk in range(3):
                        v = cf * vm[jj, kk]
                        if v != 0:
                            V6[(br*3+jj)*6 + (bc*3+kk), start+r] += v
            for bi, cf in c_blks:
                br, bc = bi // 2, bi % 2
                for kk in range(3):
                    for ii in range(3):
                        v = cf * wm[kk, ii]
                        if v != 0:
                            W6[(bc*3+kk)*6 + (br*3+ii), start+r] += v

    # P1 = (A00+A11)(B00+B11) -> C00, C11
    add_term(0,
             [(0,1),(3,1)],
             [(0,1),(3,1)],
             [(0,1),(3,1)])
    # P2 = (A10+A11)B00 -> C10(+), C11(-)
    add_term(r3,
             [(2,1),(3,1)],
             [(0,1)],
             [(2,1),(3,-1)])
    # P3 = A00(B01-B11) -> C01(+), C11(+)
    add_term(2*r3,
             [(0,1)],
             [(1,1),(3,-1)],
             [(1,1),(3,1)])
    # P4 = A11(B10-B00) -> C00(+), C10(+)
    add_term(3*r3,
             [(3,1)],
             [(2,1),(0,-1)],
             [(0,1),(2,1)])
    # P5 = (A00+A01)B11 -> C01(+), C00(-)
    add_term(4*r3,
             [(0,1),(1,1)],
             [(3,1)],
             [(1,1),(0,-1)])
    # P6 = (A10-A00)(B00+B01) -> C11(+)
    add_term(5*r3,
             [(2,1),(0,-1)],
             [(0,1),(1,1)],
             [(3,1)])
    # P7 = (A01-A11)(B10+B11) -> C00(+)
    add_term(6*r3,
             [(1,1),(3,-1)],
             [(2,1),(3,1)],
             [(0,1)])

    # Verify Strassen decomposition
    target = np.ones((6, 6, 6))
    def check(U, V, W):
        Ug = U.reshape(6, 6, -1)
        Vg = V.reshape(6, 6, -1)
        Wg = W.reshape(6, 6, -1)
        rec = np.einsum('ijr,jkr,kir->ijk', Ug, Vg, Wg)
        return np.allclose(rec, target, atol=1e-9)

    if check(U6, V6, W6):
        best_U, best_V, best_W = U6, V6, W6
    else:
        best_U, best_V, best_W = U_s, V_s, W_s

    # Attempt to reduce rank by identifying linearly dependent rank terms.
    # The idea: if we can find a subset of columns of U6,V6,W6 that still spans
    # the same tensor, we can drop the redundant columns.
    if check(U6, V6, W6) and time.time() < deadline - 10.0:
        try:
            # Find rank of U6, V6, W6
            rU = np.linalg.matrix_rank(U6, tol=1e-10)
            rV = np.linalg.matrix_rank(V6, tol=1e-10)
            rW = np.linalg.matrix_rank(W6, tol=1e-10)
            min_r = min(rU, rV, rW)

            if min_r < total_rank:
                keep = list(range(total_rank))
                # Greedy removal: try to remove one term at a time
                improved = True
                while improved:
                    improved = False
                    for r in list(keep):
                        if len(keep) <= min_r:
                            break
                        new_keep = [k for k in keep if k != r]
                        if not new_keep:
                            continue
                        U_t = U6[:, new_keep]
                        V_t = V6[:, new_keep]
                        W_t = W6[:, new_keep]
                        if check(U_t, V_t, W_t):
                            keep = new_keep
                            best_U, best_V, best_W = U_t, V_t, W_t
                            improved = True
                            break
                    if time.time() > deadline - 5.0:
                        break
        except Exception:
            pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
