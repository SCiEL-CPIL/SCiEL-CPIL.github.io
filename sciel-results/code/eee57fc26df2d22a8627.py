import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def _build_strassen_56(n, m, p):
    """Build the rank-56 Strassen-based decomposition for 2x2 * 2x16."""
    num_blocks = p // 2  # 8 blocks of 2x2
    R = num_blocks * 7   # rank 56

    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    # Strassen coefficients for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h), M2 = (c+d)(e), M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7, C01 = M3 + M5
    # C10 = M2 + M4, C11 = M1 - M2 + M3 + M6

    # U coefficients: for each Strassen term, which A entries contribute
    # A flat index: 0=a, 1=b, 2=c, 3=d
    U_coeffs = [
        {0: 1, 3: 1},   # M1: (a+d)
        {2: 1, 3: 1},   # M2: (c+d)
        {0: 1},          # M3: a
        {3: 1},          # M4: d
        {0: 1, 1: 1},   # M5: (a+b)
        {2: 1, 0: -1},  # M6: (c-a)
        {1: 1, 3: -1}   # M7: (b-d)
    ]

    # V coefficients: for each Strassen term, which B entries contribute
    # B flat index within block: 0=e, 1=f, 2=g, 3=h
    V_coeffs = [
        {0: 1, 3: 1},   # M1: (e+h)
        {0: 1},          # M2: e
        {1: 1, 3: -1},  # M3: (f-h)
        {2: 1, 0: -1},  # M4: (g-e)
        {3: 1},          # M5: h
        {0: 1, 1: 1},   # M6: (e+f)
        {2: 1, 3: 1}    # M7: (g+h)
    ]

    # W contributions: for each Strassen term, which C entries it contributes to
    # C flat index within block: 0=C00, 1=C01, 2=C10, 3=C11
    W_contribs = [
        {0: 1, 3: 1},   # M1 -> C00, C11
        {2: 1, 3: -1},  # M2 -> C10, C11
        {1: 1, 3: 1},   # M3 -> C01, C11
        {0: 1, 2: 1},   # M4 -> C00, C10
        {0: -1, 1: 1},  # M5 -> C00, C01
        {3: 1},          # M6 -> C11
        {0: 1}           # M7 -> C00
    ]

    for q in range(num_blocks):
        R_offset = q * 7
        # B indices for this block: columns 2q and 2q+1
        b_00 = 0 * p + 2 * q      # B[0, 2q] = e
        b_01 = 0 * p + (2 * q + 1)  # B[0, 2q+1] = f
        b_10 = 1 * p + 2 * q      # B[1, 2q] = g
        b_11 = 1 * p + (2 * q + 1)  # B[1, 2q+1] = h
        b_map = {0: b_00, 1: b_01, 2: b_10, 3: b_11}

        # C indices: W[k*n+i] where k=col, i=row
        # C[0, 2q], C[0, 2q+1], C[1, 2q], C[1, 2q+1]
        c_00 = (2 * q) * n + 0      # C[0, 2q]
        c_01 = (2 * q + 1) * n + 0  # C[0, 2q+1]
        c_10 = (2 * q) * n + 1      # C[1, 2q]
        c_11 = (2 * q + 1) * n + 1  # C[1, 2q+1]
        c_map = {0: c_00, 1: c_01, 2: c_10, 3: c_11}

        for s in range(7):
            r = R_offset + s

            for a_idx, coeff in U_coeffs[s].items():
                U[a_idx, r] = coeff

            for local_idx, coeff in V_coeffs[s].items():
                V[b_map[local_idx], r] = coeff

            for local_out_idx, coeff in W_contribs[s].items():
                W[c_map[local_out_idx], r] = coeff

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_strassen_56(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
