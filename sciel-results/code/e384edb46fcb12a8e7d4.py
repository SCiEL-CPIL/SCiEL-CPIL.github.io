"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 108.0

    # Block partition: J -> {0,1},{2,3},{4}; K -> {0,1},{2,3},{4,5}
    # Six 2x2x2 Strassen blocks (rank 7 each) + three 2x1x2 standard blocks (rank 4 each)
    # Total rank = 6*7 + 3*4 = 54
    R = 54
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    c = 0

    def put(u_pairs, v_pairs, w_pairs):
        nonlocal c
        for idx, val in u_pairs:
            U[idx, c] += val
        for idx, val in v_pairs:
            V[idx, c] += val
        for idx, val in w_pairs:
            W[idx, c] += val
        c += 1

    # Strassen 2x2x2 for rows i in {0,1}, cols j in {ja,jb}, out k in {ka,kb}
    def strassen(ja, jb, ka, kb):
        ua0 = 0 * m + ja
        ub0 = 0 * m + jb
        ua1 = 1 * m + ja
        ub1 = 1 * m + jb
        va_ka = ja * p + ka
        va_kb = ja * p + kb
        vb_ka = jb * p + ka
        vb_kb = jb * p + kb
        wk0_0 = ka * n + 0
        wk0_1 = ka * n + 1
        wk1_0 = kb * n + 0
        wk1_1 = kb * n + 1

        put([(ua0, 1), (ub1, 1)], [(va_ka, 1), (vb_kb, 1)], [(wk0_0, 1), (wk1_1, 1)])
        put([(ua1, 1), (ub1, 1)], [(va_ka, 1)], [(wk0_1, 1), (wk1_1, -1)])
        put([(ua0, 1)], [(va_kb, 1), (vb_kb, -1)], [(wk1_0, 1), (wk1_1, 1)])
        put([(ub1, 1)], [(vb_ka, 1), (va_ka, -1)], [(wk0_0, 1), (wk0_1, 1)])
        put([(ua0, 1), (ub0, 1)], [(vb_kb, 1)], [(wk0_0, -1), (wk1_0, 1)])
        put([(ua1, 1), (ua0, -1)], [(va_ka, 1), (va_kb, 1)], [(wk1_1, 1)])
        put([(ub0, 1), (ub1, -1)], [(vb_ka, 1), (vb_kb, 1)], [(wk0_0, 1)])

    # Six Strassen blocks
    for (ja, jb) in [(0, 1), (2, 3)]:
        for (ka, kb) in [(0, 1), (2, 3), (4, 5)]:
            strassen(ja, jb, ka, kb)

    # Three 2x1x2 standard blocks for j=4
    for k in range(p):
        for i in range(n):
            U[i * m + 4, c] = 1.0
            V[4 * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
            c += 1

    # Vectorized verification
    valid = True
    if c == R:
        # Check all (i,j,k) triples
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(U[i * m + j, :c] * V[j * p + k, :c], W[k * n + i, :c]))
                    max_err = max(max_err, abs(s - 1.0))
        valid = max_err < 1e-10

    if not valid or c != R:
        # Fallback: trivial rank 60
        Uf = np.zeros((n * m, n * m * p), dtype=float)
        Vf = np.zeros((m * p, n * m * p), dtype=float)
        Wf = np.zeros((p * n, n * m * p), dtype=float)
        cc = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i * m + j, cc] = 1.0
                    Vf[j * p + k, cc] = 1.0
                    Wf[k * n + i, cc] = 1.0
                    cc += 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    return {"U": U[:, :R].tolist(), "V": V[:, :R].tolist(), "W": W[:, :R].tolist()}
# EVOLVE-BLOCK-END
