"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]  # (16, rank)
    V = [[0] * rank for _ in range(m * p)]  # (24, rank)
    W = [[0] * rank for _ in range(p * n)]  # (24, rank)
    
    col = 0
    
    def add_strassen_term(a_block, b_block, c_block, u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        bi, bj = a_block
        bk, bl = b_block
        ci, cl = c_block
        
        for (dr, dc), coeff in u_coeffs.items():
            a_row = bi * 2 + dr
            a_col = bj * 2 + dc
            u_idx = a_row * m + a_col
            U[u_idx][col] += coeff
        
        for (dr, dc), coeff in v_coeffs.items():
            b_row = bk * 2 + dr
            b_col = bl * 2 + dc
            v_idx = b_row * p + b_col
            V[v_idx][col] += coeff
        
        for (dr, dc), coeff in w_coeffs.items():
            c_row = ci * 2 + dr
            c_col = cl * 2 + dc
            w_idx = c_col * n + c_row
            W[w_idx][col] += coeff
        
        col += 1
    
    strassen_terms = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,0):1,(0,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
    ]
    
    for ci in range(2):
        for cl in range(3):
            for u_c, v_c, w_c in strassen_terms:
                add_strassen_term((ci, 0), (0, cl), (ci, cl), u_c, v_c, w_c)
            for u_c, v_c, w_c in strassen_terms:
                add_strassen_term((ci, 1), (1, cl), (ci, cl), u_c, v_c, w_c)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
