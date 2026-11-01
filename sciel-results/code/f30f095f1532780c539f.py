import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank1(u_idx_val, v_idx_val, w_idx_val):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_idx_val:
            U[idx][col] += val
        for idx, val in v_idx_val:
            V[idx][col] += val
        for idx, val in w_idx_val:
            W[idx][col] += val
        col += 1

    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        a_r1 = a_r0 + 1
        a_c1 = a_c0 + 1
        b_r1 = b_r0 + 1
        b_c1 = b_c0 + 1
        c_r1 = c_r0 + 1
        c_c1 = c_c0 + 1
        
        # M1
        add_rank1(
            [(a_r0*m + a_c0, 1), (a_r1*m + a_c1, 1)],
            [(b_r0*p + b_c0, 1), (b_r1*p + b_c1, 1)],
            [(c_r0*n + c_c0, 1), (c_r1*n + c_c1, 1)]
        )
        # M2
        add_rank1(
            [(a_r1*m + a_c0, 1), (a_r1*m + a_c1, 1)],
            [(b_r0*p + b_c0, 1)],
            [(c_r1*n + c_c0, 1)]
        )
        # M3
        add_rank1(
            [(a_r0*m + a_c0, 1)],
            [(b_r0*p + b_c1, 1), (b_r1*p + b_c1, -1)],
            [(c_r0*n + c_c1, 1)]
        )
        # M4
        add_rank1(
            [(a_r1*m + a_c1, 1)],
            [(b_r1*p + b_c0, 1), (b_r0*p + b_c0, -1)],
            [(c_r0*n + c_c0, 1), (c_r1*n + c_c0, 1)]
        )
        # M5
        add_rank1(
            [(a_r0*m + a_c0, 1), (a_r0*m + a_c1, 1)],
            [(b_r1*p + b_c1, 1)],
            [(c_r0*n + c_c0, -1), (c_r0*n + c_c1, 1)]
        )
        # M6
        add_rank1(
            [(a_r1*m + a_c0, 1), (a_r0*m + a_c0, -1)],
            [(b_r0*p + b_c0, 1), (b_r0*p + b_c1, 1)],
            [(c_r1*n + c_c1, 1)]
        )
        # M7
        add_rank1(
            [(a_r0*m + a_c1, 1), (a_r1*m + a_c1, -1)],
            [(b_r1*p + b_c0, 1), (b_r1*p + b_c1, 1)],
            [(c_r0*n + c_c0, 1)]
        )

    def add_schoolbook_221(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        a_r1 = a_r0 + 1
        a_c1 = a_c0 + 1
        b_r1 = b_r0 + 1
        c_r1 = c_r0 + 1
        
        # S1
        add_rank1(
            [(a_r0*m + a_c0, 1)],
            [(b_r0*p + b_c0, 1)],
            [(c_r0*n + c_c0, 1)]
        )
        # S2
        add_rank1(
            [(a_r0*m + a_c1, 1)],
            [(b_r1*p + b_c0, 1)],
            [(c_r0*n + c_c0, 1)]
        )
        # S3
        add_rank1(
            [(a_r1*m + a_c0, 1)],
            [(b_r0*p + b_c0, 1)],
            [(c_r1*n + c_c0, 1)]
        )
        # S4
        add_rank1(
            [(a_r1*m + a_c1, 1)],
            [(b_r1*p + b_c0, 1)],
            [(c_r1*n + c_c0, 1)]
        )

    # 1. M_2,4,2 part
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # 2. M_2,4,3 part
    add_strassen(0, 0, 0, 2, 0, 2)
    add_schoolbook_221(0, 0, 0, 4, 0, 4)
    
    add_strassen(0, 2, 2, 2, 0, 2)
    add_schoolbook_221(0, 2, 2, 4, 0, 4)
    
    return {"U": U, "V": V, "W": W}
