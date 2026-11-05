import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank-26 decomposition for 3x3x3, used twice for 3x3x6
    rank = 52
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a rank-1 term
    def add_term(r, u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U[u_idx, r] += u_val
        V[v_idx, r] += v_val
        W[w_idx, r] += w_val
    
    # Build one 3x3x3 block (26 terms)
    # Strassen products for 2x2 block (rows/cols 0,1)
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b10)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b10
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Schoolbook for remaining 19 terms
    
    # For block offset: col_offset in B and C
    for block in range(2):
        base = block * 26
        # Column offset for B and C
        col_off = block * 3
        
        # Strassen M1 = (a00+a11)(b00+b11)
        r = base + 0
        add_term(r, 0, 1, col_off, 1, col_off * n + 0, 1)      # a00*b00 -> C[0,0]
        add_term(r, 0, 1, 1 + col_off, 1, 1 * n + 0, 1)        # a00*b11 -> C[0,1]
        add_term(r, 3, 1, col_off, 1, col_off * n + 1, 1)      # a11*b00 -> C[1,0]
        add_term(r, 3, 1, 1 + col_off, 1, 1 * n + 1, 1)        # a11*b11 -> C[1,1]
        
        # Strassen M2 = (a10+a11)b00
        r = base + 1
        add_term(r, 3, 1, col_off, 1, col_off * n + 1, 1)      # a10*b00 -> C[1,0]
        add_term(r, 6, 1, col_off, 1, col_off * n + 1, 1)      # a11*b00 -> C[1,0]
        
        # Strassen M3 = a00(b01-b10)
        r = base + 2
        add_term(r, 0, 1, 1 + col_off, 1, 1 * n + 0, 1)        # a00*b01 -> C[0,1]
        add_term(r, 0, 1, col_off, -1, col_off * n + 0, -1)    # a00*(-b10) -> C[0,0]... 
