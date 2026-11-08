import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # The minimum known rank for 2x4x4 matrix multiplication is 28.
    # This is achieved by applying Strassen's algorithm (rank 7) to four 2x2x2 sub-problems.
    # 4 * 7 = 28.
    rank = 28

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen 2x2x2 terms
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),       # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),       # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),       # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),       # M7
    ]

    def apply_strassen(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_off + c)
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (b_row_off + r) * p + (b_col_off + c)
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (c_col_off + c) * n + r
                W[g_idx, col] += coeff

    # 1. Top-Left C block (cols 0-1) from A-left (cols 0-1) * B-top-left (rows 0-1, cols 0-1)
    apply_strassen(0, 0, 0, 0, 0)
    
    # 2. Top-Left C block (cols 0-1) from A-right (cols 2-3) * B-bottom-left (rows 2-3, cols 0-1)
    apply_strassen(2, 2, 0, 0, 7)
    
    # 3. Top-Right C block (cols 2-3) from A-left (cols 0-1) * B-top-right (rows 0-1, cols 2-3)
    apply_strassen(0, 0, 2, 2, 14)
    
    # 4. Top-Right C block (cols 2-3) from A-right (cols 2-3) * B-bottom-right (rows 2-3, cols 2-3)
    apply_strassen(2, 2, 2, 2, 21)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)

    # Verification
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B

    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc

    if not np.allclose(C, Cf, atol=1e-10):
        print("Mismatch")
    else:
        print("OK rank", U.shape[1])
