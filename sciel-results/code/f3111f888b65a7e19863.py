"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    terms = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        terms.append((u_coeffs, v_coeffs, w_coeffs))

    def add_strassen_222(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        a00 = (a_row_offset)*m + a_col_offset
        a01 = (a_row_offset)*m + (a_col_offset+1)
        a10 = (a_row_offset+1)*m + a_col_offset
        a11 = (a_row_offset+1)*m + (a_col_offset+1)
        
        b00 = (b_row_offset)*p + b_col_offset
        b01 = (b_row_offset)*p + (b_col_offset+1)
        b10 = (b_row_offset+1)*p + b_col_offset
        b11 = (b_row_offset+1)*p + (b_col_offset+1)
        
        c00 = (c_row_offset)*n + c_col_offset
        c01 = (c_row_offset)*n + (c_col_offset+1)
        c10 = (c_row_offset+1)*n + c_col_offset
        c11 = (c_row_offset+1)*n + (c_col_offset+1)
        
        # M1
        add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        # M2
        add_term({a10: 1, a11: 1}, {b00: 1}, {c10: 1, c11: 1})
        # M3
        add_term({a00: 1}, {b01: 1, b11: -1}, {c01: 1, c11: -1})
        # M4
        add_term({a11: 1}, {b10: 1, b00: -1}, {c00: -1, c10: 1})
        # M5
        add_term({a00: 1, a01: 1}, {b11: 1}, {c01: 1, c11: -1})
        # M6
        add_term({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c00: 1, c01: -1})
        # M7
        add_term({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: -1, c01: -1, c10: 1})

    # A1 (2x2) * B1 (2x7)
    # B1 split into 3 blocks of 2x2 and 1 block of 2x1
    add_strassen_222(0, 0, 0, 0, 0, 0)
    add_strassen_222(0, 0, 0, 2, 2, 0)
    add_strassen_222(0, 0, 0, 4, 4, 0)
    
    # A1 (2x2) * B14 (2x1) at k=6
    add_term({0*m+0: 1}, {0*p+6: 1}, {6*n+0: 1})
    add_term({0*m+1: 1}, {1*p+6: 1}, {6*n+0: 1})
    add_term({1*m+0: 1}, {0*p+6: 1}, {6*n+1: 1})
    add_term({1*m+1: 1}, {1*p+6: 1}, {6*n+1: 1})
    
    # A2 (2x1) * B2 (1x7)
    for i in range(2):
        for k in range(7):
            u_idx = i*m + 2
            v_idx = 2*p + k
            w_idx = k*n + i
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
            
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r, (u_c, v_c, w_c) in enumerate(terms):
        for idx, val in u_c.items():
            U[idx][r] = val
        for idx, val in v_c.items():
            V[idx][r] = val
        for idx, val in w_c.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
