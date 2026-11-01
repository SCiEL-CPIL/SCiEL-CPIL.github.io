def construct():
    n, m, p = 2, 3, 4
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 12 x 22
    W = [[0] * rank for _ in range(p * n)]  # 8 x 22
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :] -> C[:, 0:4]
    # This is <2, 1, 4>. Rank 8.
    for i in range(n):
        for k in range(p):
            # A index: i * m + 0
            # B index: 0 * p + k
            # C index: k * n + i
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A[:, 1:3] * B[1:3, 0:2] -> C[:, 0:2]
    # This is <2, 2, 2> using Strassen. Rank 7.
    # A submatrix: rows 0-1, cols 1-2.
    # B submatrix: rows 1-2, cols 0-1.
    # C submatrix: rows 0-1, cols 0-1.
    
    # Strassen coefficients for 2x2 * 2x2
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # Map local indices to global
    # Local A (i, j) -> Global A (i, j+1)
    # Local B (j, k) -> Global B (j+1, k)
    # Local C (i, k) -> Global C (i, k)
    
    def add_strassen_term(col_idx, a_coeffs, b_coeffs, c_coeffs):
        # a_coeffs: list of (local_i, local_j, coeff)
        # b_coeffs: list of (local_j, local_k, coeff)
        # c_coeffs: list of (local_i, local_k, coeff)
        for li, lj, ca in a_coeffs:
            gi = li
            gj = lj + 1 # offset by 1
            U[gi * m + gj][col_idx] += ca
        for lj, lk, cb in b_coeffs:
            gj = lj + 1 # offset by 1
            gk = lk
            V[gj * p + gk][col_idx] += cb
        for li, lk, cc in c_coeffs:
            gi = li
            gk = lk
            W[gk * n + gi][col_idx] += cc

    # m1
    add_strassen_term(col, [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
    col += 1
    # m2
    add_strassen_term(col, [(1,0,1), (1,1,1)], [(0,0,1), (1,0,0)], [(1,0,1), (1,1,1)])
    # Wait, m2 = (a10+a11)b00.
    # a10 is local (1,0), a11 is local (1,1).
    # b00 is local (0,0).
    # Result C: c10 += m2, c11 += m2? No.
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    # c10 (local 1,0) gets m2.
    # c11 (local 1,1) gets -m2? No, m1-m2...
    # Let's re-verify Strassen formula.
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # So:
    # m1 contributes to C00, C11
    # m2 contributes to C10, C11 (with -1)
    # m3 contributes to C01, C11
    # m4 contributes to C00, C10
    # m5 contributes to C00 (with -1), C01
    # m6 contributes to C11
    # m7 contributes to C00
    
    # Let's re-do the add_strassen_term calls with correct C contributions.
    
    # Reset col for Part 2? No, col is currently 8.
    # m1
    add_strassen_term(col, [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
    col += 1
    # m2
    add_strassen_term(col, [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
    col += 1
    # m3
    add_strassen_term(col, [(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
    col += 1
    # m4
    add_strassen_term(col, [(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
    col += 1
    # m5
    add_strassen_term(col, [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
    col += 1
    # m6
    add_strassen_term(col, [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
    col += 1
    # m7
    add_strassen_term(col, [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    col += 1

    # Part 3: A[:, 1:3] * B[1:3, 2:4] -> C[:, 2:4]
    # This is <2, 2, 2> using Strassen. Rank 7.
    # A submatrix: rows 0-1, cols 1-2. (Same A)
    # B submatrix: rows 1-2, cols 2-3.
    # C submatrix: rows 0-1, cols 2-3.
    
    # Map local indices to global
    # Local A (i, j) -> Global A (i, j+1)
    # Local B (j, k) -> Global B (j+1, k+2)
    # Local C (i, k) -> Global C (i, k+2)
    
    def add_strassen_term_p3(col_idx, a_coeffs, b_coeffs, c_coeffs):
        for li, lj, ca in a_coeffs:
            gi = li
            gj = lj + 1
            U[gi * m + gj][col_idx] += ca
        for lj, lk, cb in b_coeffs:
            gj = lj + 1
            gk = lk + 2
            V[gj * p + gk][col_idx] += cb
        for li, lk, cc in c_coeffs:
            gi = li
            gk = lk + 2
            W[gk * n + gi][col_idx] += cc

    # m1
    add_strassen_term_p3(col, [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
    col += 1
    # m2
    add_strassen_term_p3(col, [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
    col += 1
    # m3
    add_strassen_term_p3(col, [(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
    col += 1
    # m4
    add_strassen_term_p3(col, [(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
    col += 1
    # m5
    add_strassen_term_p3(col, [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
    col += 1
    # m6
    add_strassen_term_p3(col, [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
    col += 1
    # m7
    add_strassen_term_p3(col, [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    col += 1

    return {"U": U, "V": V, "W": W}
