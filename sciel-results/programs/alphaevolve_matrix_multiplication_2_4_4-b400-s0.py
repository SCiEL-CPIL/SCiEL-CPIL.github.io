import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen on 2x2 blocks
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec, col_idx):
        # u_vec: dict or list of (index, coeff)
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
    
    # We need to map local 2x2 indices to global indices.
    # For A (2x4):
    # A_1 (cols 0,1): 
    #   Local (0,0) -> Global (0,0) -> U[0*4+0]
    #   Local (0,1) -> Global (0,1) -> U[0*4+1]
    #   Local (1,0) -> Global (1,0) -> U[1*4+0]
    #   Local (1,1) -> Global (1,1) -> U[1*4+1]
    # A_2 (cols 2,3):
    #   Local (0,0) -> Global (0,2) -> U[0*4+2]
    #   Local (0,1) -> Global (0,3) -> U[0*4+3]
    #   Local (1,0) -> Global (1,2) -> U[1*4+2]
    #   Local (1,1) -> Global (1,3) -> U[1*4+3]
    
    # For B (4x4):
    # B_L,1 (rows 0,1, cols 0,1):
    #   Local (0,0) -> Global (0,0) -> V[0*4+0]
    #   Local (0,1) -> Global (0,1) -> V[0*4+1]
    #   Local (1,0) -> Global (1,0) -> V[1*4+0]
    #   Local (1,1) -> Global (1,1) -> V[1*4+1]
    
    # B_L,2 (rows 2,3, cols 0,1):
    #   Local (0,0) -> Global (2,0) -> V[2*4+0]
    #   Local (0,1) -> Global (2,1) -> V[2*4+1]
    #   Local (1,0) -> Global (3,0) -> V[3*4+0]
    #   Local (1,1) -> Global (3,1) -> V[3*4+1]
    
    # B_R,1 (rows 0,1, cols 2,3):
    #   Local (0,0) -> Global (0,2) -> V[0*4+2]
    #   Local (0,1) -> Global (0,3) -> V[0*4+3]
    #   Local (1,0) -> Global (1,2) -> V[1*4+2]
    #   Local (1,1) -> Global (1,3) -> V[1*4+3]
    
    # B_R,2 (rows 2,3, cols 2,3):
    #   Local (0,0) -> Global (2,2) -> V[2*4+2]
    #   Local (0,1) -> Global (2,3) -> V[2*4+3]
    #   Local (1,0) -> Global (3,2) -> V[3*4+2]
    #   Local (1,1) -> Global (3,3) -> V[3*4+3]
    
    # For C (2x4):
    # C_L (cols 0,1):
    #   Local (0,0) -> Global (0,0) -> W[0*2+0] (k=0, i=0)
    #   Local (0,1) -> Global (0,1) -> W[1*2+0] (k=1, i=0)
    #   Local (1,0) -> Global (1,0) -> W[0*2+1] (k=0, i=1)
    #   Local (1,1) -> Global (1,1) -> W[1*2+1] (k=1, i=1)
    
    # C_R (cols 2,3):
    #   Local (0,0) -> Global (0,2) -> W[2*2+0] (k=2, i=0)
    #   Local (0,1) -> Global (0,3) -> W[3*2+0] (k=3, i=0)
    #   Local (1,0) -> Global (1,2) -> W[2*2+1] (k=2, i=1)
    #   Local (1,1) -> Global (1,3) -> W[3*2+1] (k=3, i=1)
    
    # Define Strassen terms for a generic 2x2 product P = XY
    # Returns list of 7 tuples: (u_coeff_dict, v_coeff_dict, w_coeff_dict)
    def strassen_terms():
        # X indices: 00, 01, 10, 11
        # Y indices: 00, 01, 10, 11
        # P indices: 00, 01, 10, 11
        
        # M1 = (X00+X11)(Y00+Y11)
        # P00 += M1, P11 -= M1? No, P11 = M1 - M2 + M3 + M6
        # P00 = M1 + M4 - M5 + M7
        # P01 = M3 + M5
        # P10 = M2 + M4
        # P11 = M1 - M2 + M3 + M6
        
        terms = []
        
        # M1
        u = {0: 1, 3: 1} # X00, X11
        v = {0: 1, 3: 1} # Y00, Y11
        w = {0: 1, 3: 1} # P00, P11
        terms.append((u, v, w))
        
        # M2
        u = {2: 1, 3: 1} # X10, X11
        v = {0: 1} # Y00
        w = {2: -1, 3: 1} # P10, P11 (P10=M2+M4, P11=M1-M2+...)
        # Wait, P10 coeff of M2 is 1. P11 coeff of M2 is -1.
        w = {2: 1, 3: -1}
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
        # a_offset: 0 for A_1 (cols 0,1), 1 for A_2 (cols 2,3)
        # b_row_offset: 0 for top (rows 0,1), 1 for bottom (rows 2,3)
        # b_col_offset: 0 for left (cols 0,1), 1 for right (cols 2,3)
        # c_col_offset: 0 for left (cols 0,1), 1 for right (cols 2,3)
        
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
    # A_1: offset 0. B_L,1: row_off 0, col_off 0. C_L: col_off 0.
    apply_strassen(0, 0, 0, 0, 0)
    
    # 2. C_L, Part 2: A_2 * B_L,2
    # A_2: offset 1. B_L,2: row_off 1, col_off 0. C_L: col_off 0.
    apply_strassen(1, 1, 0, 0, 7)
    
    # 3. C_R, Part 1: A_1 * B_R,1
    # A_1: offset 0. B_R,1: row_off 0, col_off 1. C_R: col_off 1.
    apply_strassen(0, 0, 1, 1, 14)
    
    # 4. C_R, Part 2: A_2 * B_R,2
    # A_2: offset 1. B_R,2: row_off 1, col_off 1. C_R: col_off 1.
    apply_strassen(1, 1, 1, 1, 21)
    
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
