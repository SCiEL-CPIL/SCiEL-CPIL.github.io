import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    # Rank-28 Strassen decomposition for 2x4 * 4x4 matrix multiplication.
    # A is 2x4, B is 4x4, C = A @ B is 2x4.
    #
    # The tensor T_{(i,j),(j,k)}(i,k) = delta_{i,k via j} has rank 28.
    # We use a block-wise Strassen approach:
    #   Split A into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
    #   Split B into four 2x2 blocks: B_TL, B_TR, B_BL, B_BR.
    #   C_L (cols 0-1) = A_L*B_TL + A_R*B_BL
    #   C_R (cols 2-3) = A_L*B_TR + A_R*B_BR
    #   Each 2x2*2x2 multiplication uses Strassen (rank 7).
    #   Total rank = 4 * 7 = 28.

    # Strassen columns for 2x2 * 2x2 -> 2x2
    # Each column: dict {U_part: [(local_idx, coeff)], V_part: [(local_idx, coeff)], W_part: [(local_idx, coeff)]}
    # local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)

    S = [
        # M1: (A00+A11)(B00+B11) -> C00+C11
        ([[(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]],),
        # M2: (A10+A11)B00 -> C10-C11
        ([[(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]],),
        # M3: A00(B01-B11) -> C01+C11
        ([[(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]],),
        # M4: A11(B10-B00) -> C00+C10
        ([[(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]],),
        # M5: (A00+A01)B11 -> -C00+C01
        ([[(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]],),
        # M6: (A10-A00)(B00+B01) -> C11
        ([[(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]],),
        # M7: (A01-A11)(B10+B11) -> C00
        ([[(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]],),
    ]

    n, m, p = DIMENSIONS
    R = 28
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    # Mapping functions
    def mA(li, off):
        r, c = li // 2, li % 2
        return r * m + (off * 2 + c)

    def mB(li, ro, co):
        r, c = li // 2, li % 2
        return (ro * 2 + r) * p + (co * 2 + c)

    def mC(li, co):
        r, c = li // 2, li % 2
        return (co * 2 + c) * n + r

    # Four blocks: (A_offset, B_row_off, B_col_off, C_col_off, col_base)
    blocks = [
        (0, 0, 0, 0, 0),   # A_L * B_TL -> C_L
        (1, 1, 0, 0, 7),   # A_R * B_BL -> C_L
        (0, 0, 1, 1, 14),  # A_L * B_TR -> C_R
        (1, 1, 1, 1, 21),  # A_R * B_BR -> C_R
    ]

    for a_off, b_ro, b_co, c_co, base in blocks:
        for j, (parts,) in enumerate(S):
            ci = base + j
            u_entries, v_entries, w_entries = parts
            for li, cf in u_entries:
                U[mA(li, a_off), ci] += cf
            for li, cf in v_entries:
                V[mB(li, b_ro, b_co), ci] += cf
            for li, cf in w_entries:
                W[mC(li, c_co), ci] += cf

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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
