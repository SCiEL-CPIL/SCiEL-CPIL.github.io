"""Matrix multiplication 4x4x8 with rank-98 Strassen-based decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _build_2x2x2_strassen():
    """Rank-7 Strassen decomposition for 2x2x2.
    U index: i*2+j, V index: j*2+k, W index: k*2+i.
    Returns U,V,W each (4,7)."""
    terms = [
        ([1, 0, 0, -1], [1, 0, 0, 1],   [1, 0, 0, 1]),   # M1 = (a-d)(e+h)
        ([0, 0, 1, 1],   [1, 0, 0, 0],   [0, 0, 1, -1]),  # M2 = (c+d)e
        ([1, 0, 0, 0],   [0, 1, 0, -1],  [0, 1, 0, 1]),   # M3 = a(f-h)
        ([0, 0, 0, 1],   [-1, 0, 1, 0],  [1, 0, 1, 0]),   # M4 = d(g-e)
        ([1, 1, 0, 0],   [0, 0, 0, 1],   [-1, 1, 0, 0]),  # M5 = (a+b)h
        ([0, 0, 1, -1],  [0, 1, 0, 1],   [0, 0, 0, 1]),   # M6 = (c-d)(f+h)
        ([1, 0, -1, 0],  [1, 0, 1, 0],   [1, 0, 0, 0]),   # M7 = (a-c)(e+g)
    ]
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u; V[:, r] = v; W[:, r] = w
    return U, V, W


def _build_4x4x4_strassen2():
    """Rank-49 decomposition for 4x4x4 via recursive Strassen.
    U index: i*4+j, V index: j*4+k, W index: k*4+i.
    Returns U,V,W each (16,49)."""
    U2, V2, W2 = _build_2x2x2_strassen()

    # Block-level Strassen operations (left coeffs on A blocks, right on B blocks)
    block_ops = [
        ([1, 0, 0, -1], [1, 0, 0, 1]),    # M1 = (A00-A11)(B00+B11)
        ([0, 0, 1, 1],   [1, 0, 0, 0]),    # M2 = (A10+A11)B00
        ([1, 0, 0, 0],   [0, 1, 0, -1]),   # M3 = A00(B01-B11)
        ([0, 0, 0, 1],   [0, 0, 1, -1]),   # M4 = A11(B10-B00)
        ([1, 1, 0, 0],   [0, 0, 0, 1]),    # M5 = (A00+A01)B11
        ([0, 0, 1, -1],  [0, 1, 0, 1]),    # M6 = (A10-A11)(B01+B11)
        ([1, 0, -1, 0],  [1, 0, 1, 0]),    # M7 = (A00-A10)(B00+B10)
    ]
    # Output block combos: C00, C01, C10, C11 (coeffs over M1..M7)
    c_combos = [
        [1, 0, 0, 1, -1, 0, 1],    # C00 = M1+M4-M5+M7
        [0, 0, 1, 0, 1, 0, 0],     # C01 = M3+M5
        [0, 1, 0, 1, 0, 0, 0],     # C10 = M2+M4
        [1, -1, 1, 0, 0, 1, 0],    # C11 = M1-M2+M3+M6
    ]

    U = np.zeros((16, 49)); V = np.zeros((16, 49)); W = np.zeros((16, 49))
    # For each block op b (0..6) and each sub-Strassen term t (0..6), rank col = b*7+t
    for b in range(7):
        lc, rc = block_ops[b]
        for t in range(7):
            col = b * 7 + t
            # U factor: U2[ib*2+jb, t] * lc[ib*2+jb] for full index i*4+j
            # i*4+j = ib*4 + (jb*2 + i2) = ib*4 + jb*2 + i2
            for ib in range(2):
                for jb in range(2):
                    for i2 in range(2):
                        j2 = t  # placeholder, loop below
            # Build U column vector of length 16
            for i2 in range(2):
                for j2 in range(2):
                    # U2 index = i2*2+j2
                    for ib in range(2):
                        for jb in range(2):
                            a_idx = ib * 2 + jb
                            u2val = U2[i2 * 2 + j2, t]
                            if u2val == 0:
                                continue
                            full_u = ib * 4 + jb * 2 + i2 * 0 + (j2)  # need correct mapping
                            # full index i*4+j where i = ib*2+i2, j = jb*2+j2
                            full_i = ib * 2 + i2
                            full_j = jb * 2 + j2
                            U[full_i * 4 + full_j, col] += u2val * lc[a_idx]
            # V column vector of length 16
            for j2 in range(2):
                for k2 in range(2):
                    for jb in range(2):
                        for kb in range(2):
                            b_idx = jb * 2 + kb
                            v2val = V2[j2 * 2 + k2, t]
                            if v2val == 0:
                                continue
                            full_j = jb * 2 + j2
                            full_k = kb * 2 + k2
                            V[full_j * 4 + full_k, col] += v2val * rc[b_idx]
            # W column vector of length 16
            for i2 in range(2):
                for k2 in range(2):
                    for ib in range(2):
                        for kb in range(2):
                            c_idx = kb * 2 + ib  # C block index = kb*2+ib
                            w2val = W2[k2 * 2 + i2, t]
                            if w2val == 0:
                                continue
                            full_k = kb * 2 + k2
                            full_i = ib * 2 + i2
                            W[full_k * 4 + full_i, col] += w2val * c_combos[c_idx][b]
    return U, V, W


def _verify(U, V, W, rank):
    """Verify that for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = float(np.dot(U[u_idx], V[v_idx] * W[w_idx]))
                if abs(val - 1.0) > 1e-6:
                    return False
    return True


def _schoolbook():
    n, m, p = DIMENSIONS
    nm = n * m; mp = m * p; pn = p * n
    rank = n * m * p
    U = np.zeros((nm, rank)); V = np.zeros((mp, rank)); W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    nm = n * m; mp = m * p; pn = p * n
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Build rank-98 decomposition: two 4x4x4 Strassen blocks for the 8 columns
    U4, V4, W4 = _build_4x4x4_strassen2()
    # U4: (16,49), V4: (16,49), W4: (16,49) with indices i*4+j, j*4+k, k*4+i
    # For 4x4x8: V index j*8+k, W index k*4+i.
    # Split k in {0..7} into two blocks of 4: kb in {0,1}, k_local in {0..3}
    # Block 0: k = 0..3, Block 1: k = 4..7
    # U: (16, 98), V: (32, 98), W: (32, 98)
    rank = 98
    U = np.zeros((nm, rank)); V = np.zeros((mp, rank)); W = np.zeros((pn, rank))

    for kb in range(2):  # two k-blocks
        base = kb * 49
        # U: same for both blocks (rows i*4+j, 16 rows)
        U[:, base:base+49] = U4
        # V: rows j*8+k, k = kb*4 + k_local. U4's V rows are j*4+k_local
        for j in range(4):
            for k_local in range(4):
                v4_row = j * 4 + k_local
                v_full_row = j * 8 + (kb * 4 + k_local)
                V[v_full_row, base:base+49] = V4[v4_row]
        # W: rows k*4+i, k = kb*4 + k_local. U4's W rows are k_local*4+i
        for k_local in range(4):
            for i in range(4):
                w4_row = k_local * 4 + i
                w_full_row = (kb * 4 + k_local) * 4 + i
                W[w_full_row, base:base+49] = W4[w4_row]

    if time.time() - start_time > deadline:
        Us, Vs, Ws = _schoolbook()
        return {"U": [[float(x) for x in row] for row in Us],
                "V": [[float(x) for x in row] for row in Vs],
                "W": [[float(x) for x in row] for row in Ws]}

    # Verify the rank-98 decomposition
    if _verify(U, V, W, rank):
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}

    # Fallback: try schoolbook
    Us, Vs, Ws = _schoolbook()
    if _verify(Us, Vs, Ws, Us.shape[1]):
        return {"U": [[float(x) for x in row] for row in Us],
                "V": [[float(x) for x in row] for row in Vs],
                "W": [[float(x) for x in row] for row in Ws]}

    # Last resort: return schoolbook anyway
    return {"U": [[float(x) for x in row] for row in Us],
            "V": [[float(x) for x in row] for row in Vs],
            "W": [[float(x) for x in row] for row in Ws]}
# EVOLVE-BLOCK-END
