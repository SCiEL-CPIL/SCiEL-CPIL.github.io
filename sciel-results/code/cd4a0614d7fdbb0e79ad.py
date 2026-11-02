import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Uses Strassen's algorithm on 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 terms
    # X (2x2) * Y (2x2) = P (2x2)
    # Indices: 0:00, 1:01, 2:10, 3:11
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    def map_u_a(side, local_idx):
        if side == 'L':
            return local_idx
        else:
            return 2 + local_idx

    def map_v_b(block, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if block == 'TL':
            return r * 4 + c
        elif block == 'TR':
            return r * 4 + (2 + c)
        elif block == 'BL':
            return (2 + r) * 4 + c
        elif block == 'BR':
            return (2 + r) * 4 + (2 + c)

    def map_w_c(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if side == 'L':
            return c * 2 + r
        else:
            return (2 + c) * 2 + r

    def apply_strassen(a_side, b_block, c_side, start_col):
        for t in strassen_terms:
            u_local, v_local, w_local = t
            col = start_col
            for idx, coeff in u_local:
                U[map_u_a(a_side, idx), col] += coeff
            for idx, coeff in v_local:
                V[map_v_b(b_block, idx), col] += coeff
            for idx, coeff in w_local:
                W[map_w_c(c_side, idx), col] += coeff
            start_col += 1
        return start_col

    # 1. A_L * B_TL -> C_L
    col = apply_strassen('L', 'TL', 'L', 0)
    # 2. A_R * B_BL -> C_L
    col = apply_strassen('R', 'BL', 'L', col)
    # 3. A_L * B_TR -> C_R
    col = apply_strassen('L', 'TR', 'R', col)
    # 4. A_R * B_BR -> C_R
    col = apply_strassen('R', 'BR', 'R', col)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
