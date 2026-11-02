import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We know rank is 50
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i):
        nonlocal col
        # Local indices i, j, k in {0, 1}
        # U index: (u_offset_i + i) * m + (u_offset_j + j)
        # V index: (v_offset_j + j) * p + (v_offset_k + k)
        # W index: (w_offset_k + k) * n + (w_offset_i + i)
        
        # Strassen terms definitions
        # M1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
        # M2: (A10+A11)B00 -> C10(+1), C11(-1)
        # M3: A00(B01-B11) -> C01(+1), C11(+1)
        # M4: A11(B10-B00) -> C00(+1), C10(+1)
        # M5: (A00+A01)B11 -> C00(-1), C01(+1)
        # M6: (A10-A00)(B00+B01) -> C11(+1)
        # M7: (A01-A11)(B10+B11) -> C00(+1)
        
        terms = [
            # (U_vals, V_vals, W_vals)
            # U_vals: list of (local_j, coeff) for A elements? No, local (i,j)
            # Let's just use dicts for local indices
            {
                'u': [(0, 1), (1, 1)], # A00, A11
                'v': [(0, 1), (1, 1)], # B00, B11
                'w': [(0, 1), (1, 1)]  # C00, C11
            },
            {
                'u': [(1, 1), (1, 1)], # A10, A11
                'v': [(0, 1)],         # B00
                'w': [(1, 1), (1, -1)] # C10, C11
            },
            {
                'u': [(0, 1)],         # A00
                'v': [(1, 1), (1, -1)],# B01, B11
                'w': [(1, 1), (1, 1)]  # C01, C11
            },
            {
                'u': [(1, 1)],         # A11
                'v': [(1, 1), (0, -1)],# B10, B00
                'w': [(0, 1), (1, 1)]  # C00, C10
            },
            {
                'u': [(0, 1), (0, 1)], # A00, A01
                'v': [(1, 1)],         # B11
                'w': [(0, -1), (1, 1)] # C00, C01
            },
            {
                'u': [(1, 1), (0, -1)],# A10, A00
                'v': [(0, 1), (1, 1)], # B00, B01
                'w': [(1, 1)]          # C11
            },
            {
                'u': [(0, 1), (1, -1)],# A01, A11
                'v': [(1, 1), (1, 1)], # B10, B11
                'w': [(0, 1)]          # C00
            }
        ]
        
        for t in terms:
            for (li, coeff_u) in t['u']:
                ui = (u_offset_i + li) * m + (u_offset_j + t['u'][0][0] if False else 0) # Wait, need to track j for U
                # My 'u' list above has (i, coeff) but I need j too.
                # Let's refine the term structure.
                pass
            
    # Let's rewrite the helper to be precise with indices.
    def add_strassen_block_precise(u_i_off, u_j_off, v_j_off, v_k_off, w_k_off, w_i_off):
        nonlocal col
        # Terms: (u_entries, v_entries, w_entries)
        # u_entries: [(i, j, coeff), ...]
        # v_entries: [(j, k, coeff), ...]
        # w_entries: [(k, i, coeff), ...]
        
        terms = [
            # M1
            ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
            # M2
            ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),
            # M3
            ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
            # M4
            ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 1, 1)]),
            # M5
            ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (1, 0, 1)]),
            # M6
            ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
            # M7
            ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)])
        ]
        
        for u_terms, v_terms, w_terms in terms:
            for (li, lj, cu) in u_terms:
                idx_u = (u_i_off + li) * m + (u_j_off + lj)
                U[idx_u][col] += cu
            for (lj, lk, cv) in v_terms:
                idx_v = (v_j_off + lj) * p + (v_k_off + lk)
                V[idx_v][col] += cv
            for (lk, li, cw) in w_terms:
                idx_w = (w_k_off + lk) * n + (w_i_off + li)
                W[idx_w][col] += cw
            col += 1

    # Helper for 2x2x1 schoolbook
    def add_schoolbook_221(u_i_off, u_j_off, v_j_off, v_k_off, w_k_off, w_i_off):
        nonlocal col
        # A is 2x2, B is 2x1. C is 2x1.
        # C0 = A00*B00 + A01*B10
        # C1 = A10*B00 + A11*B10
        
        # Term 1: A00 * B00 -> C0
        U[(u_i_off + 0) * m + (u_j_off + 0)][col] = 1
        V[(v_j_off + 0) * p + (v_k_off + 0)][col] = 1
        W[(w_k_off + 0) * n + (w_i_off + 0)][col] = 1
        col += 1
        
        # Term 2: A01 * B10 -> C0
        U[(u_i_off + 0) * m + (u_j_off + 1)][col] = 1
        V[(v_j_off + 1) * p + (v_k_off + 0)][col] = 1
        W[(w_k_off + 0) * n + (w_i_off + 0)][col] = 1
        col += 1
        
        # Term 3: A10 * B00 -> C1
        U[(u_i_off + 1) * m + (u_j_off + 0)][col] = 1
        V[(v_j_off + 0) * p + (v_k_off + 0)][col] = 1
        W[(w_k_off + 0) * n + (w_i_off + 1)][col] = 1
        col += 1
        
        # Term 4: A11 * B10 -> C1
        U[(u_i_off + 1) * m + (u_j_off + 1)][col] = 1
        V[(v_j_off + 1) * p + (v_k_off + 0)][col] = 1
        W[(w_k_off + 0) * n + (w_i_off + 1)][col] = 1
        col += 1

    # Block 1: A1 B1a1 (2x2x2)
    # A1: i=0,1, j=0,1. B1a1: j=0,1, k=0,1.
    add_strassen_block_precise(0, 0, 0, 0, 0, 0)
    
    # Block 2: A1 B1a2 (2x2x2)
    # A1: i=0,1, j=0,1. B1a2: j=0,1, k=2,3.
    add_strassen_block_precise(0, 0, 0, 2, 2, 0)
    
    # Block 3: A1 B1b1 (2x2x2)
    # A1: i=0,1, j=0,1. B1b1: j=0,1, k=4,5.
    add_strassen_block_precise(0, 0, 0, 4, 4, 0)
    
    # Block 4: A1 B1b2 (2x2x1)
    # A1: i=0,1, j=0,1. B1b2: j=0,1, k=6.
    add_schoolbook_221(0, 0, 0, 6, 6, 0)
    
    # Block 5: A2 B2a1 (2x2x2)
    # A2: i=0,1, j=2,3. B2a1: j=2,3, k=0,1.
    add_strassen_block_precise(0, 2, 2, 0, 0, 0)
    
    # Block 6: A2 B2a2 (2x2x2)
    # A2: i=0,1, j=2,3. B2a2: j=2,3, k=2,3.
    add_strassen_block_precise(0, 2, 2, 2, 2, 0)
    
    # Block 7: A2 B2b1 (2x2x2)
    # A2: i=0,1, j=2,3. B2b1: j=2,3, k=4,5.
    add_strassen_block_precise(0, 2, 2, 4, 4, 0)
    
    # Block 8: A2 B2b2 (2x2x1)
    # A2: i=0,1, j=2,3. B2b2: j=2,3, k=6.
    add_schoolbook_221(0, 2, 2, 6, 6, 0)
    
    return {"U": U, "V": V, "W": W}
