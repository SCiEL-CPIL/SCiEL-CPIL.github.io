import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We will define the 4 Strassen blocks.
    # Each block uses 7 scalar multiplications.
    # Total rank = 4 * 7 = 28.
    
    # Helper to add a term to U, V, W
    def add_term(u_entries, v_entries, w_entries, r):
        for idx, val in u_entries:
            U[idx][r] += val
        for idx, val in v_entries:
            V[idx][r] += val
        for idx, val in w_entries:
            W[idx][r] += val

    # Strassen terms for 2x2 matrix multiplication X*Y
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Each term is a tuple (u_entries, v_entries, w_entries)
    # Entries are lists of (local_index, value)
    
    strassen_terms = [
        # M1 = (X00 + X11) * (Y00 + Y11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (X10 + X11) * Y00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = X00 * (Y01 - Y11)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = X11 * (Y10 - Y00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (X00 + X01) * Y11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (X10 - X00) * (Y00 + Y01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (X01 - X11) * (Y10 + Y11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    def apply_strassen(a_col_off, b_row_off, b_col_off, c_col_off, base_rank):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            r = base_rank + t_idx
            
            u_entries = []
            for loc_idx, val in u_loc:
                lr = loc_idx // 2
                lc = loc_idx % 2
                gr = lr
                gc = a_col_off * 2 + lc
                g_idx = gr * m + gc
                u_entries.append((g_idx, val))
            
            v_entries = []
            for loc_idx, val in v_loc:
                lr = loc_idx // 2
                lc = loc_idx % 2
                gr = b_row_off * 2 + lr
                gc = b_col_off * 2 + lc
                g_idx = gr * p + gc
                v_entries.append((g_idx, val))
            
            w_entries = []
            for loc_idx, val in w_loc:
                lr = loc_idx // 2
                lc = loc_idx % 2
                gr = lr
                gc = c_col_off * 2 + lc
                # W index convention: W[k*n + i, r] where k is col, i is row
                # In our mapping, 'gc' is the column of C (k), 'gr' is the row of C (i)
                g_idx = gc * n + gr
                w_entries.append((g_idx, val))
            
            add_term(u_entries, v_entries, w_entries, r)

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    apply_strassen(0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    apply_strassen(1, 1, 0, 0, 7)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    apply_strassen(0, 0, 1, 1, 14)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    apply_strassen(1, 1, 1, 1, 21)

    return {"U": U, "V": V, "W": W}
