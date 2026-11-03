def construct():
    n, m, p = 3, 3, 4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
        col += 1

    # 1. A_TL (2x2) * B_TL (2x2) -> C_TL (2x2)
    # Strassen Rank 7
    # Local indices for A_TL: (0,0), (0,1), (1,0), (1,1) -> Global U: 0, 1, 3, 4
    # Local indices for B_TL: (0,0), (0,1), (1,0), (1,1) -> Global V: 0, 1, 3, 4 (j=0,1; k=0,1)
    # Local indices for C_TL: (0,0), (0,1), (1,0), (1,1) -> Global W: 0, 3, 1, 4 (k=0,1; i=0,1)
    
    # Strassen Terms for 2x2x2
    # M1: (A00+A11)(B00+B11) -> C00, C11
    add_term([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (4, 1)])
    
    # M2: (A10+A11)B00 -> C10
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1)])
    
    # M3: A00(B01-B11) -> C01
    add_term([(0, 1)], [(1, 1), (4, -1)], [(3, 1)])
    
    # M4: A11(B10-B00) -> C00, C10
    add_term([(4, 1)], [(3, 1), (0, -1)], [(0, 1), (1, 1)])
    
    # M5: (A00+A01)B11 -> C00, C01
    add_term([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (3, 1)])
    
    # M6: (A10-A00)(B00+B01) -> C11
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(4, 1)])
    
    # M7: (A01-A11)(B10+B11) -> C00, C11
    add_term([(1, 1), (4, -1)], [(3, 1), (4, 1)], [(0, 1), (4, 1)])
    
    # 2. A_TR (2x1) * B_BL (1x2) -> C_TL (2x2) part
    # A_TR: (0,2), (1,2) -> U: 2, 5
    # B_BL: (2,0), (2,1) -> V: 8, 9 (j=2, k=0,1)
    # C_TL: (0,0), (0,1), (1,0), (1,1) -> W: 0, 3, 1, 4
    # Schoolbook:
    # A02 B20 -> C00
    add_term([(2, 1)], [(8, 1)], [(0, 1)])
    # A02 B21 -> C01
    add_term([(2, 1)], [(9, 1)], [(3, 1)])
    # A12 B20 -> C10
    add_term([(5, 1)], [(8, 1)], [(1, 1)])
    # A12 B21 -> C11
    add_term([(5, 1)], [(9, 1)], [(4, 1)])
    
    # 3. A_TL (2x2) * B_TR (2x2) -> C_TR (2x2)
    # A_TL: U 0,1,3,4
    # B_TR: (0,2), (0,3), (1,2), (1,3) -> V: 2, 3, 5, 6 (j=0,1; k=2,3)
    # C_TR: (0,2), (0,3), (1,2), (1,3) -> W: 6, 7, 9, 10 (k=2,3; i=0,1)
    
    # Strassen Terms for 2x2x2
    # M1: (A00+A11)(B00+B11) -> C00, C11 (Local C indices 0,1 -> Global 6, 10)
    add_term([(0, 1), (4, 1)], [(2, 1), (6, 1)], [(6, 1), (10, 1)])
    
    # M2: (A10+A11)B00 -> C10 (Local 1 -> Global 9)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(9, 1)])
    
    # M3: A00(B01-B11) -> C01 (Local 0 -> Global 7)
    add_term([(0, 1)], [(3, 1), (6, -1)], [(7, 1)])
    
    # M4: A11(B10-B00) -> C00, C10 (Local 0,1 -> Global 6, 9)
    add_term([(4, 1)], [(5, 1), (2, -1)], [(6, 1), (9, 1)])
    
    # M5: (A00+A01)B11 -> C00, C01 (Local 0,1 -> Global 6, 7)
    add_term([(0, 1), (1, 1)], [(6, 1)], [(6, -1), (7, 1)])
    
    # M6: (A10-A00)(B00+B01) -> C11 (Local 1 -> Global 10)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(10, 1)])
    
    # M7: (A01-A11)(B10+B11) -> C00, C11 (Local 0,1 -> Global 6, 10)
    add_term([(1, 1), (4, -1)], [(5, 1), (6, 1)], [(6, 1), (10, 1)])
    
    # 4. A_TR (2x1) * B_BR (1x2) -> C_TR (2x2) part
    # A_TR: U 2, 5
    # B_BR: (2,2), (2,3) -> V: 10, 11
    # C_TR: W 6, 7, 9, 10
    
    add_term([(2, 1)], [(10, 1)], [(6, 1)])
    add_term([(2, 1)], [(11, 1)], [(7, 1)])
    add_term([(5, 1)], [(10, 1)], [(9, 1)])
    add_term([(5, 1)], [(11, 1)], [(10, 1)])
    
    # 5. A_BL (1x2) * B_TL (2x2) -> C_BL (1x2)
    # A_BL: (2,0), (2,1) -> U: 6, 7
    # B_TL: V 0, 1, 3, 4
    # C_BL: (2,0), (2,1) -> W: 12, 13 (k=0,1; i=2)
    
    add_term([(6, 1)], [(0, 1)], [(12, 1)])
    add_term([(6, 1)], [(1, 1)], [(13, 1)])
    add_term([(7, 1)], [(3, 1)], [(12, 1)])
    add_term([(7, 1)], [(4, 1)], [(13, 1)])
    
    # 6. A_BR (1x1) * B_BL (1x2) -> C_BL (1x2) part
    # A_BR: (2,2) -> U: 8
    # B_BL: V 8, 9
    # C_BL: W 12, 13
    
    add_term([(8, 1)], [(8, 1)], [(12, 1)])
    add_term([(8, 1)], [(9, 1)], [(13, 1)])
    
    # 7. A_BL (1x2) * B_TR (2x2) -> C_BR (1x2)
    # A_BL: U 6, 7
    # B_TR: V 2, 3, 5, 6
    # C_BR: (2,2), (2,3) -> W: 16, 17 (k=2,3; i=2)
    
    add_term([(6, 1)], [(2, 1)], [(16, 1)])
    add_term([(6, 1)], [(3, 1)], [(17, 1)])
    add_term([(7, 1)], [(5, 1)], [(16, 1)])
    add_term([(7, 1)], [(6, 1)], [(17, 1)])
    
    # 8. A_BR (1x1) * B_BR (1x2) -> C_BR (1x2) part
    # A_BR: U 8
    # B_BR: V 10, 11
    # C_BR: W 16, 17
    
    add_term([(8, 1)], [(10, 1)], [(16, 1)])
    add_term([(8, 1)], [(11, 1)], [(17, 1)])
    
    return {"U": U, "V": V, "W": W}
