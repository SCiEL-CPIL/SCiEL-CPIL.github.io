"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size 2x2, grid 3x3
    # Rank = 27 (grid terms) * 7 (Strassen rank) = 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen steps for 2x2
    # Each step: (a_terms, b_terms, c_terms)
    # a_terms: list of (coeff, u, w)
    # b_terms: list of (coeff, w, v)
    # c_terms: list of (coeff, u, v)
    
    steps = [
        # M1 = (A00 + A11)(B00 + B11) -> C00 +1, C11 +1
        ([(1, 0, 0), (1, 1, 1)], [(1, 0, 0), (1, 1, 1)], [(1, 0, 0), (1, 1, 1)]),
        # M2 = (A10 + A11)B00 -> C10 +1, C11 -1
        ([(1, 1, 0), (1, 1, 1)], [(1, 0, 0)], [(1, 1, 0), (-1, 1, 1)]),
        # M3 = A00(B01 - B11) -> C01 +1, C11 +1
        ([(1, 0, 0)], [(1, 0, 1), (-1, 1, 1)], [(1, 0, 1), (1, 1, 1)]),
        # M4 = A11(B10 - B00) -> C00 +1, C10 +1
        ([(1, 1, 1)], [(1, 1, 0), (-1, 0, 0)], [(1, 0, 0), (1, 1, 0)]),
        # M5 = (A00 + A01)B11 -> C00 -1, C01 +1
        ([(1, 0, 0), (1, 0, 1)], [(1, 1, 1)], [(-1, 0, 0), (1, 0, 1)]),
        # M6 = (A10 - A00)(B00 + B01) -> C11 +1
        ([(1, 1, 0), (-1, 0, 0)], [(1, 0, 0), (1, 0, 1)], [(1, 1, 1)]),
        # M7 = (A01 - A11)(B10 + B11) -> C00 +1
        ([(1, 0, 1), (-1, 1, 1)], [(1, 1, 0), (1, 1, 1)], [(1, 0, 0)]),
    ]
    
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for a_terms, b_terms, c_terms in steps:
                    for u in range(2):
                        for v in range(2):
                            for w in range(2):
                                ca = 0
                                for coeff, ua, wa in a_terms:
                                    if ua == u and wa == w:
                                        ca += coeff
                                cb = 0
                                for coeff, wb, vb in b_terms:
                                    if wb == w and vb == v:
                                        cb += coeff
                                
                                if ca == 0 or cb == 0:
                                    continue
                                    
                                for cc, uc, vc in c_terms:
                                    if uc == u and vc == v:
                                        # Global indices
                                        row_A = 2 * i + u
                                        col_A = 2 * k + w
                                        row_B = 2 * k + w
                                        col_B = 2 * j + v
                                        row_C = 2 * i + u
                                        col_C = 2 * j + v
                                        
                                        idx_A = row_A * m + col_A
                                        idx_B = row_B * p + col_B
                                        idx_C = col_C * n + row_C
                                        
                                        coeff = ca * cb * cc
                                        
                                        U[idx_A][col] += coeff
                                        V[idx_B][col] += coeff
                                        W[idx_C][col] += coeff
                                        
                                        col += 1
                                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
