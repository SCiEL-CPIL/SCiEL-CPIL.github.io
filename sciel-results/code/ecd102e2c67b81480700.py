import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 decomposition terms
    # Indices: 00, 01, 10, 11
    # M1 = (X00+X11)(Y00+Y11) -> P00 += 1, P11 += 1
    # M2 = (X10+X11)Y00       -> P10 += 1, P11 -= 1
    # M3 = X00(Y01-Y11)       -> P01 += 1, P11 += 1
    # M4 = X11(Y10-Y00)       -> P00 += 1, P10 += 1
    # M5 = (X00+X01)Y11       -> P00 -= 1, P01 += 1
    # M6 = (X10-X00)(Y00+Y01) -> P11 += 1
    # M7 = (X01-X11)(Y10+Y11) -> P00 += 1
    
    terms = [
        # u, v, w
        ([0, 3], [0, 3], [0, 3]), # M1: X00+X11, Y00+Y11 -> P00+P11
        ([2, 3], [0], [2, 3]),    # M2: X10+X11, Y00 -> P10+P11
        ([0], [1, 3], [1, 3]),    # M3: X00, Y01-Y11 -> P01+P11 (Coeffs: 1, 1)
        ([3], [2, 0], [0, 2]),    # M4: X11, Y10-Y00 -> P00+P10 (Coeffs: 1, 1)
        ([0, 1], [3], [0, 1]),    # M5: X00+X01, Y11 -> P00+P01 (Coeffs: -1, 1)
        ([2, 0], [0, 1], [3]),    # M6: X10-X00, Y00+Y01 -> P11 (Coeff: 1)
        ([1, 3], [2, 3], [0])     # M7: X01-X11, Y10+Y11 -> P00 (Coeff: 1)
    ]
    
    # Correcting the signs in the terms list above to match the equations:
    # M1: u=[0,3], v=[0,3], w=[0:1, 3:1]
    # M2: u=[2,3], v=[0], w=[2:1, 3:-1]
    # M3: u=[0], v=[1:1, 3:-1], w=[1:1, 3:1]
    # M4: u=[3], v=[2:1, 0:-1], w=[0:1, 2:1]
    # M5: u=[0:1, 1:1], v=[3], w=[0:-1, 1:1]
    # M6: u=[2:1, 0:-1], v=[0:1, 1:1], w=[3:1]
    # M7: u=[1:1, 3:-1], v=[2:1, 3:1], w=[0:1]
    
    def add_term(u_indices, v_indices, w_indices, col_idx):
        # u_indices, v_indices, w_indices are lists of (index, coeff)
        for idx, coeff in u_indices:
            U[idx, col_idx] += coeff
        for idx, coeff in v_indices:
            V[idx, col_idx] += coeff
        for idx, coeff in w_indices:
            W[idx, col_idx] += coeff

    # Define the 7 terms with explicit coefficients
    # Local indices: 0=00, 1=01, 2=10, 3=11
    s_terms = [
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
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]

    def apply_block(a_off, b_row_off, b_col_off, c_col_off, start_col):
        for i, (u_loc, v_loc, w_loc) in enumerate(s_terms):
            col = start_col + i
            
            u_global = []
            for loc_idx, coeff in u_loc:
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_off * 2 + c
                g_idx = g_r * 4 + g_c
                u_global.append((g_idx, coeff))
            
            v_global = []
            for loc_idx, coeff in v_loc:
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_off * 2 + r
                g_c = b_col_off * 2 + c
                g_idx = g_r * 4 + g_c
                v_global.append((g_idx, coeff))
            
            w_global = []
            for loc_idx, coeff in w_loc:
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_off * 2 + c
                # W[k*n + i] -> k is col, i is row
                g_idx = g_c * 2 + g_r
                w_global.append((g_idx, coeff))
                
            add_term(u_global, v_global, w_global, col)

    # 1. A1 * BTL -> CL
    apply_block(0, 0, 0, 0, 0)
    # 2. A2 * BBL -> CL
    apply_block(1, 1, 0, 0, 7)
    # 3. A1 * BTR -> CR
    apply_block(0, 0, 1, 1, 14)
    # 4. A2 * BBR -> CR
    apply_block(1, 1, 1, 1, 21)

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
