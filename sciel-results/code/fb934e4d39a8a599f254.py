def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for 2x2 multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We map these to the 2x4 by 4x8 case.
    # A is 2x4, B is 4x8.
    # We split A into A1 (cols 0,1) and A2 (cols 2,3).
    # We split B into 4 blocks of 2x2? No, B is 4x8.
    # B rows 0,1 correspond to A col 0,1?
    # A = [A1, A2] where A1 is 2x2, A2 is 2x2.
    # B = [[B11, B12], [B21, B22]] where Bij is 2x4.
    # C = A1 B11 + A2 B21 (for cols 0-3) and A1 B12 + A2 B22 (for cols 4-7).
    
    # Actually, simpler:
    # For each of the 4 2x2 blocks of B (splitting cols into pairs),
    # and for each of the 2 2x2 blocks of A (splitting cols into pairs),
    # we have a 2x2 by 2x2 multiplication?
    # No.
    # A is 2x4. B is 4x8.
    # C_{ik} = sum_{j=0}^3 A_{ij} B_{jk}.
    
    # Let's iterate over the 4 pairs of columns in B.
    # For k_pair in 0..3:
    #   B_block is B[:, 2*k_pair : 2*k_pair+2] which is 4x2.
    #   We need A * B_block.
    #   A is 2x4.
    #   Split A into A_left (cols 0,1) and A_right (cols 2,3).
    #   Split B_block into B_top (rows 0,1) and B_bot (rows 2,3). Both 2x2.
    #   C_block = A_left B_top + A_right B_bot.
    
    # So for each k_pair (4 times), we compute:
    # 1. A_left B_top (2x2 by 2x2) -> 7 terms
    # 2. A_right B_bot (2x2 by 2x2) -> 7 terms
    # Total 14 terms per k_pair.
    # 4 * 14 = 56 terms.
    
    # This matches the logic.
    
    # Strassen coefficients for one 2x2 multiplication
    # U: A entries, V: B entries, W: C entries
    # Indices:
    # A: 0,1 (row 0), 2,3 (row 1) -> local (0,0), (0,1), (1,0), (1,1)
    # B: 0,1 (row 0), 2,3 (row 1) -> local (0,0), (0,1), (1,0), (1,1)
    # C: 0,1 (row 0), 2,3 (row 1) -> local (0,0), (0,1), (1,0), (1,1)
    
    # We define the 7 terms for 2x2 multiplication
    # Each term is (U_coefs, V_coefs, W_coefs)
    # U_coefs: list of (local_row, local_col, val)
    # V_coefs: list of (local_row, local_col, val)
    # W_coefs: list of (local_row, local_col, val)
    
    terms_2x2 = [
        # P1
        {
            "U": [(0,0,1), (1,1,1)],
            "V": [(0,0,1), (1,1,1)],
            "W": [(0,0,1), (1,1,1)]
        },
        # P2
        {
            "U": [(1,0,1), (1,1,1)],
            "V": [(0,0,1)],
            "W": [(1,0,1), (1,1,-1)]
        },
        # P3
        {
            "U": [(0,0,1)],
            "V": [(0,1,1), (1,1,-1)],
            "W": [(0,1,1), (1,1,1)]
        },
        # P4
        {
            "U": [(1,1,1)],
            "V": [(1,0,1), (0,0,-1)],
            "W": [(0,0,1), (1,0,1)]
        },
        # P5
        {
            "U": [(0,0,1), (0,1,1)],
            "V": [(1,1,1)],
            "W": [(0,0,-1), (0,1,1)]
        },
        # P6
        {
            "U": [(1,0,1), (0,0,-1)],
            "V": [(0,0,1), (0,1,1)],
            "W": [(1,1,1)]
        },
        # P7
        {
            "U": [(0,1,1), (1,1,-1)],
            "V": [(1,0,1), (1,1,1)],
            "W": [(0,0,1)]
        },
    ]
    
    # We have 4 k-pairs (columns of B)
    # For each k-pair, we have 2 A-blocks (left/right)
    # A-left corresponds to A cols 0,1 and B rows 0,1
    # A-right corresponds to A cols 2,3 and B rows 2,3
    
    for k_pair in range(4):
        # Process A-left * B-top
        # A-left: rows 0,1; cols 0,1
        # B-top: rows 0,1; cols 2*k_pair, 2*k_pair+1
        for term in terms_2x2:
            for (ri, ci, val) in term["U"]:
                # A global index: row ri, col ci
                U[ri * 4 + ci][col] += val
            
            for (ri, ci, val) in term["V"]:
                # B global index: row ri, col 2*k_pair + ci
                V[ri * 8 + (2 * k_pair + ci)][col] += val
            
            for (ri, ci, val) in term["W"]:
                # C global index: row ri, col 2*k_pair + ci
                # W convention: W[k*n + i, r] where k is col of C, i is row of C?
                # Task says: W[k*n+i, r]. n=2.
                # So W index = k*2 + i.
                W[(2 * k_pair + ci) * 2 + ri][col] += val
            
            col += 1
            
        # Process A-right * B-bot
        # A-right: rows 0,1; cols 2,3
        # B-bot: rows 2,3; cols 2*k_pair, 2*k_pair+1
        for term in terms_2x2:
            for (ri, ci, val) in term["U"]:
                # A global index: row ri, col 2 + ci
                U[ri * 4 + (2 + ci)][col] += val
            
            for (ri, ci, val) in term["V"]:
                # B global index: row 2 + ri, col 2*k_pair + ci
                V[(2 + ri) * 8 + (2 * k_pair + ci)][col] += val
            
            for (ri, ci, val) in term["W"]:
                # C global index: row ri, col 2*k_pair + ci
                W[(2 * k_pair + ci) * 2 + ri][col] += val
            
            col += 1

    return {"U": U, "V": V, "W": W}
