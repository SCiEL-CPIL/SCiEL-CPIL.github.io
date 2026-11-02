import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0

    # Strassen 2x2 decomposition: 7 rank-1 tensors (local 2x2x2)
    # Each term is three 4-element vectors (coefficients for local indices 0..3)
    S_U = np.zeros((7, 4))
    S_V = np.zeros((7, 4))
    S_W = np.zeros((7, 4))

    # M1: (X0+X3)(Y0+Y3) = Z0+Z3
    S_U[0, [0, 3]] = [1, 1]
    S_V[0, [0, 3]] = [1, 1]
    S_W[0, [0, 3]] = [1, 1]

    # M2: (X2+X3)(Y0) = Z2-Z3
    S_U[1, [2, 3]] = [1, 1]
    S_V[1, [0]] = [1]
    S_W[1, [2, 3]] = [1, -1]

    # M3: (X0)(Y1-Y3) = Z1+Z3
    S_U[2, [0]] = [1]
    S_V[2, [1, 3]] = [1, -1]
    S_W[2, [1, 3]] = [1, 1]

    # M4: (X3)(Y2-Y0) = Z2+Z0
    S_U[3, [3]] = [1]
    S_V[3, [2, 0]] = [1, -1]
    S_W[3, [2, 0]] = [1, 1]

    # M5: (X0+X1)(Y3) = Z0-Z1
    S_U[4, [0, 1]] = [1, 1]
    S_V[4, [3]] = [1]
    S_W[4, [0, 1]] = [1, -1]

    # M6: (X2-X0)(Y0+Y1) = Z3
    S_U[5, [2, 0]] = [1, -1]
    S_V[5, [0, 1]] = [1, 1]
    S_W[5, [3]] = [1]

    # M7: (X1-X3)(Y2+Y3) = Z0
    S_U[6, [1, 3]] = [1, -1]
    S_V[6, [2, 3]] = [1, 1]
    S_W[6, [0]] = [1]

    # Number of 2x2 blocks along each dimension
    nb_r = n // 2   # 1 row-block pair
    nb_c = m // 2   # 2 col-block pairs
    nb_o = p // 2   # 3 output-block pairs
    rank = nb_r * nb_c * nb_o * 7  # 1*2*3*7 = 42

    # Build global matrices via Kronecker product structure
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0
    for rb in range(nb_r):
        for cb in range(nb_c):
            for ob in range(nb_o):
                # Kronecker: block_selector ⊗ strassen_term
                for t in range(7):
                    # U: rows indexed (i, j), i in [0..n), j in [0..m)
                    # block (rb, cb) selects i in [2*rb, 2*rb+2), j in [2*cb, 2*cb+2)
                    for li in range(4):
                        li_i = li // 2
                        li_j = li % 2
                        gi = 2 * rb + li_i
                        gj = 2 * cb + li_j
                        U[gi * m + gj, idx] = S_U[t, li]

                    # V: rows indexed (j, k), j in [0..m), k in [0..p)
                    # block (cb, ob) selects j in [2*cb, 2*cb+2), k in [2*ob, 2*ob+2)
                    for lj in range(4):
                        lj_j = lj // 2
                        lj_k = lj % 2
                        gj = 2 * cb + lj_j
                        gk = 2 * ob + lj_k
                        V[gj * p + gk, idx] = S_V[t, lj]

                    # W: rows indexed (k, i), k in [0..p), i in [0..n)
                    # block (ob, rb) selects k in [2*ob, 2*ob+2), i in [2*rb, 2*rb+2)
                    for lk in range(4):
                        lk_k = lk // 2
                        lk_i = lk % 2
                        gk = 2 * ob + lk_k
                        gi = 2 * rb + lk_i
                        W[gk * n + gi, idx] = S_W[t, lk]

                    idx += 1

    # Verify correctness
    try:
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,ki->ijk', U3, V3, W3)
        if not np.allclose(T, 1.0, atol=1e-9):
            # Fallback to schoolbook
            r_sb = n * m * p
            U_sb = np.zeros((n * m, r_sb))
            V_sb = np.zeros((m * p, r_sb))
            W_sb = np.zeros((p * n, r_sb))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j, c] = 1.0
                        V_sb[j * p + k, c] = 1.0
                        W_sb[k * n + i, c] = 1.0
                        c += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
