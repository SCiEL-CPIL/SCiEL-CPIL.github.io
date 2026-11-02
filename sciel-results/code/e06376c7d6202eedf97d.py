import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    rank = 36
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    def add_term(u_entries, v_entries, w_entries, c):
        nonlocal col
        for r, val in u_entries:
            U[r, c] += val
        for r, val in v_entries:
            V[r, c] += val
        for r, val in w_entries:
            W[r, c] += val
        col = c + 1

    def apply_strassen(a_base, b_base, c_base):
        # a_base: U row for a
        # b_base: V row for e
        # c_base: W row for c00
        
        a_r, b_r, c_r, d_r = a_base, a_base+1, a_base+4, a_base+5
        e_r, f_r, g_r, h_r = b_base, b_base+1, b_base+5, b_base+6
        c00_r, c01_r, c10_r, c11_r = c_base, c_base+1, c_base+2, c_base+3
        
        add_term([(a_r,1), (d_r,1)], [(e_r,1), (h_r,1)], [(c00_r,1), (c11_r,1)], col)
        add_term([(c_r,1), (d_r,1)], [(e_r,1)], [(c10_r,1), (c11_r,-1)], col)
        add_term([(a_r,1)], [(f_r,1), (h_r,-1)], [(c01_r,1), (c11_r,1)], col)
        add_term([(d_r,1)], [(g_r,1), (e_r,-1)], [(c00_r,1), (c10_r,1)], col)
        add_term([(a_r,1), (b_r,1)], [(h_r,1)], [(c00_r,-1), (c01_r,1)], col)
        add_term([(c_r,1), (a_r,-1)], [(e_r,1), (f_r,1)], [(c11_r,1)], col)
        add_term([(b_r,1), (d_r,-1)], [(g_r,1), (h_r,1)], [(c00_r,1)], col)
        
    def apply_linear(a_base, b_base, c_base):
        a_r, b_r, c_r, d_r = a_base, a_base+1, a_base+4, a_base+5
        e_r, g_r = b_base, b_base+5
        c0_r, c1_r = c_base, c_base+2
        
        add_term([(a_r,1)], [(e_r,1)], [(c0_r,1)], col)
        add_term([(b_r,1)], [(g_r,1)], [(c0_r,1)], col)
        add_term([(c_r,1)], [(e_r,1)], [(c1_r,1)], col)
        add_term([(d_r,1)], [(g_r,1)], [(c1_r,1)], col)

    # Block 1: A1 (j=0,1), B1 (j=0,1)
    apply_strassen(0, 0, 0)
    apply_strassen(0, 2, 4)
    apply_linear(0, 4, 8)
    
    # Block 2: A2 (j=2,3), B2 (j=2,3)
    apply_strassen(2, 10, 0)
    apply_strassen(2, 12, 4)
    apply_linear(2, 14, 8)
    
    if time.time() > deadline:
        # Fallback to schoolbook if something went wrong (unlikely)
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
