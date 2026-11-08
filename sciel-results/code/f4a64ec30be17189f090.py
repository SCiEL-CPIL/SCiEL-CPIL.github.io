import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    _start_time = time.time()
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen on 2x2 blocks
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec, col_idx):
        # u_vec, v_vec, w_vec are lists of (index, coeff)
        for idx, coeff in u_vec:
            if coeff != 0:
                U[idx, col_idx] += coeff
        for idx, coeff in v_vec:
            if coeff != 0:
                V[idx, col_idx] += coeff
        for idx, coeff in w_vec:
            if coeff != 0:
                W[idx, col_idx] += coeff
            
    # Strassen 2x2 decomposition terms
    # P = X Y
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Function to apply Strassen to a block
    # a_col_off: 0 or 2 (offset in A columns)
    # b_row_off: 0 or 2 (offset in B rows)
    # b_col_off: 0 or 2 (offset in B columns)
    # c_col_off: 0 or 2 (offset in C columns)
    # start_col: starting column in U, V, W
    def apply_strassen(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            
            u_global = []
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_col_off + c
                g_idx = g_r * m + g_c
                u_global.append((g_idx, coeff))
                
            v_global = []
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_off + r
                g_c = b_col_off + c
                g_idx = g_r * p + g_c
                v_global.append((g_idx, coeff))
                
            w_global = []
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_off + c
                # W[k*n + i] where k is col, i is row
                g_idx = g_c * n + g_r
                w_global.append((g_idx, coeff))
                
            add_term(u_global, v_global, w_global, col)
            
    # 4 blocks:
    # 1. A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2] part 1
    apply_strassen(0, 0, 0, 0, 0)
    
    # 2. A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2] part 2
    apply_strassen(2, 2, 0, 0, 7)
    
    # 3. A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4] part 1
    apply_strassen(0, 0, 2, 2, 14)
    
    # 4. A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4] part 2
    apply_strassen(2, 2, 2, 2, 21)
    
    # Deadline guard
    if time.time() - _start_time > 140:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices
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
            
    if not np.allclose(C, Cf):
        print("Mismatch")
        print(C)
        print(Cf)
    else:
        print("OK rank", U.shape[1])
