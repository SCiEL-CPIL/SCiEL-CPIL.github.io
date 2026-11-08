import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 terms
    # Local indices: 0=00, 1=01, 2=10, 3=11
    s_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),          # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),          # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),          # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),          # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),          # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])           # M7
    ]

    def add_term(u_indices, v_indices, w_indices, col_idx):
        for idx, coeff in u_indices:
            U[idx, col_idx] += coeff
        for idx, coeff in v_indices:
            V[idx, col_idx] += coeff
        for idx, coeff in w_indices:
            W[idx, col_idx] += coeff

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
                g_idx = g_c * 2 + g_r
                w_global.append((g_idx, coeff))
                
            add_term(u_global, v_global, w_global, col)

    # 4 blocks of 2x2x2
    apply_block(0, 0, 0, 0, 0)   # A1 * BTL -> CL
    apply_block(1, 1, 0, 0, 7)   # A2 * BBL -> CL
    apply_block(0, 0, 1, 1, 14)  # A1 * BTR -> CR
    apply_block(1, 1, 1, 1, 21)  # A2 * BBR -> CR

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
