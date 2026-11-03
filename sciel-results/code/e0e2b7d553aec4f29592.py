import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank1(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Helper to add Strassen terms for 2x2x2 at given offsets
    def add_strassen_2x2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # a_r, a_c: row, col offset for A (2x2)
        # b_r, b_c: row, col offset for B (2x2)
        # c_r, c_c: row, col offset for C (2x2)
        
        # Indices
        # A: (a_r, a_c), (a_r, a_c+1), (a_r+1, a_c), (a_r+1, a_c+1)
        # B: (b_r, b_c), (b_r, b_c+1), (b_r+1, b_c), (b_r+1, b_c+1)
        # C: (c_r, c_c), (c_r, c_c+1), (c_r+1, c_c), (c_r+1, c_c+1)
        
        # Note: U index is i*m + j. V index is j*p + k. W index is k*n + i.
        
        # m1 = (a+b)(e+h)
        U[a_r*m + a_c][col] += 1
        U[a_r*m + a_c + 1][col] += 1
        V[b_r*p + b_c][col] += 1
        V[b_r*p + b_c + 1][col] += 1
        W[b_c*n + a_r][col] += 1
        W[b_c*n + a_r + 1][col] += 1
        W[b_c+1*n + a_r][col] += 1
        W[b_c+1*n + a_r + 1][col] += 1
        col += 1
        
        # m2 = (a-b)(e-h)
        U[a_r*m + a_c][col] += 1
        U[a_r*m + a_c + 1][col] += -1
        V[b_r*p + b_c][col] += 1
        V[b_r*p + b_c + 1][col] += -1
        W[b_c*n + a_r][col] += 1
        W[b_c*n + a_r + 1][col] += -1
        W[b_c+1*n + a_r][col] += -1
        W[b_c+1*n + a_r + 1][col] += 1
        col += 1
        
        # m3 = (c+d)(e+h)
        U[a_r+1*m + a_c][col] += 1
        U[a_r+1*m + a_c + 1][col] += 1
        V[b_r*p + b_c][col] += 1
        V[b_r*p + b_c + 1][col] += 1
        W[b_c*n + a_r][col] += 1
        W[b_c*n + a_r + 1][col] += 1
        W[b_c+1*n + a_r][col] += 1
        W[b_c+1*n + a_r + 1][col] += 1
        col += 1
        
        # m4 = (c-d)(e-h)
        U[a_r+1*m + a_c][col] += 1
        U[a_r+1*m + a_c + 1][col] += -1
        V[b_r*p + b_c][col] += 1
        V[b_r*p + b_c + 1][col] += -1
        W[b_c*n + a_r][col] += 1
        W[b_c*n + a_r + 1][col] += -1
        W[b_c+1*n + a_r][col] += -1
        W[b_c+1*n + a_r + 1][col] += 1
        col += 1
        
        # m5 = a f
        U[a_r*m + a_c][col] += 1
        V[b_r*p + b_c + 1][col] += 1
        W[b_c*n + a_r][col] += 1
        col += 1
        
        # m6 = (b+d) i
        U[a_r*m + a_c + 1][col] += 1
        U[a_r+1*m + a_c + 1][col] += 1
        V[b_r+1*p + b_c + 1][col] += 1
        W[b_c+1*n + a_r + 1][col] += 1
        col += 1
        
        # m7 = (c-a) h
        U[a_r+1*m + a_c][col] += 1
        U[a_r*m + a_c][col] += -1
        V[b_r+1*p + b_c][col] += 1
        W[b_c*n + a_r][col] += 1
        W[b_c+1*n + a_r][col] += 1
        col += 1

    # Helper to add schoolbook terms for 2x2x1
    def add_schoolbook_2x2x1(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # b_c is the column index in B (1 column)
        # c_c is the column index in C (1 column)
        
        # a e
        U[a_r*m + a_c][col] += 1
        V[b_r*p + b_c][col] += 1
        W[b_c*n + a_r][col] += 1
        col += 1
        
        # b f (where f is b_r+1, b_c)
        U[a_r*m + a_c + 1][col] += 1
        V[b_r+1*p + b_c][col] += 1
        W[b_c*n + a_r][col] += 1
        col += 1
        
        # c e
        U[a_r+1*m + a_c][col] += 1
        V[b_r*p + b_c][col] += 1
        W[b_c*n + a_r + 1][col] += 1
        col += 1
        
        # d f
        U[a_r+1*m + a_c + 1][col] += 1
        V[b_r+1*p + b_c][col] += 1
        W[b_c*n + a_r + 1][col] += 1
        col += 1

    # Process the 8 sub-products
    # A blocks:
    # A00: r=0, c=0
    # A01: r=0, c=2
    # A10: r=2, c=0
    # A11: r=2, c=2
    
    # B blocks (2x3):
    # B_TL: r=0, c=0
    # B_BL: r=2, c=0
    # B_TR: r=0, c=3
    # B_BR: r=2, c=3
    
    # C blocks (2x3):
    # C00: r=0, c=0
    # C01: r=0, c=3
    # C10: r=2, c=0
    # C11: r=2, c=3
    
    # C00 = A00 * B_TL + A01 * B_BL
    # A00 * B_TL:
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0) # Strassen for cols 0,1 of B_TL -> C00 cols 0,1
    add_schoolbook_2x2x1(0, 0, 0, 2, 0, 2) # Schoolbook for col 2 of B_TL -> C00 col 2
    add_schoolbook_2x2x1(0, 0, 2, 0, 0, 0) # Wait, B_BL is r=2, c=0..2
    # A01 * B_BL:
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0) # Strassen for cols 0,1 of B_BL -> C00 cols 0,1
    add_schoolbook_2x2x1(0, 2, 2, 2, 0, 2) # Schoolbook for col 2 of B_BL -> C00 col 2
    
    # C01 = A00 * B_TR + A01 * B_BR
    add_strassen_2x2x2(0, 0, 0, 3, 0, 3)
    add_schoolbook_2x2x1(0, 0, 0, 5, 0, 5)
    add_strassen_2x2x2(0, 2, 2, 3, 0, 3)
    add_schoolbook_2x2x1(0, 2, 2, 5, 0, 5)
    
    # C10 = A10 * B_TL + A11 * B_BL
    add_strassen_2x2x2(2, 0, 0, 0, 2, 0)
    add_schoolbook_2x2x1(2, 0, 0, 2, 2, 2)
    add_strassen_2x2x2(2, 2, 2, 0, 2, 0)
    add_schoolbook_2x2x1(2, 2, 2, 2, 2, 2)
    
    # C11 = A10 * B_TR + A11 * B_BR
    add_strassen_2x2x2(2, 0, 0, 3, 2, 3)
    add_schoolbook_2x2x1(2, 0, 0, 5, 2, 5)
    add_strassen_2x2x2(2, 2, 2, 3, 2, 3)
    add_schoolbook_2x2x1(2, 2, 2, 5, 2, 5)
    
    return {"U": U, "V": V, "W": W}
