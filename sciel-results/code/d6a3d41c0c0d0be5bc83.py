import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Rank 28 decomposition using Strassen on 2x2 blocks
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec, col_idx):
        for idx, coeff in u_vec:
            U[idx, col_idx] += coeff
        for idx, coeff in v_vec:
            V[idx, col_idx] += coeff
        for idx, coeff in w_vec:
            W[idx, col_idx] += coeff
            
    # Strassen 2x2 decomposition
    # P = XY, where X, Y are 2x2.
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    def strassen_terms():
        terms = []
        
        # M1
        u = {0: 1, 3: 1} # X00, X11
        v = {0: 1, 3: 1} # Y00, Y11
        w = {0: 1, 3: 1} # P00, P11
        terms.append((u, v, w))
        
        # M2
        u = {2: 1, 3: 1} # X10, X11
        v = {0: 1} # Y00
        w = {2: 1, 3: -1} # P10, P11
        terms.append((u, v, w))
        
        # M3
        u = {0: 1} # X00
        v = {1: 1, 3: -1} # Y01, Y11
        w = {1: 1, 3: 1} # P01, P11
        terms.append((u, v, w))
        
        # M4
        u = {3: 1} # X11
        v = {2: 1, 0: -1} # Y10, Y00
        w = {0: 1, 2: 1} # P00, P10
        terms.append((u, v, w))
        
        # M5
        u = {0: 1, 1: 1} # X00, X01
        v = {3: 1} # Y11
        w = {0: -1, 1: 1} # P00, P01
        terms.append((u, v, w))
        
        # M6
        u = {2: 1, 0: -1} # X10, X00
        v = {0: 1, 1: 1} # Y00, Y01
        w = {3: 1} # P11
        terms.append((u, v, w))
        
        # M7
        u = {1: 1, 3: -1} # X01, X11
        v = {2: 1, 3: 1} # Y10, Y11
        w = {0: 1} # P00
        terms.append((u, v, w))
        
        return terms

    base_terms = strassen_terms()
    
    # Function to apply a set of Strassen terms to specific blocks
    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            
            # Map U
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_offset * 2 + c
                g_idx = g_r * 4 + g_c
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            
            # Map V
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_offset * 2 + r
                g_c = b_col_offset * 2 + c
                g_idx = g_r * 4 + g_c
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            
            # Map W
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_offset * 2 + c
                # W[k*n + i] -> k is col, i is row
                # g_c is col, g_r is row
                g_idx = g_c * 2 + g_r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            
            add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)

    # 1. C_L, Part 1: A_1 * B_L,1
    apply_strassen(0, 0, 0, 0, 0)
    
    # 2. C_L, Part 2: A_2 * B_L,2
    apply_strassen(1, 1, 0, 0, 7)
    
    # 3. C_R, Part 1: A_1 * B_R,1
    apply_strassen(0, 0, 1, 1, 14)
    
    # 4. C_R, Part 2: A_2 * B_R,2
    apply_strassen(1, 1, 1, 1, 21)
    
    # Verify correctness
    def verify(U_mat, V_mat, W_mat):
        A = np.arange(n * m).reshape(n, m) + 1.0
        B = np.arange(m * p).reshape(m, p) + 2.0
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(U_mat.shape[1]):
                    for jj in range(m):
                        acc += U_mat[ii * m + jj, cc] * V_mat[jj * p + kk, cc] * W_mat[kk * n + ii, cc]
                Cf[ii, kk] = acc
        return np.allclose(C, Cf)
    
    if not verify(U, V, W):
        raise ValueError("Decomposition verification failed")
    
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
