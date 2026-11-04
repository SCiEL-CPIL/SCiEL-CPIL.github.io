import numpy as np

def construct():
    n, m, p = 4, 5, 6
    rank = 108
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0

    def add_term(a_idx, b_idx, c_idx, coeff=1.0):
        nonlocal col
        # a_idx: (i, j)
        # b_idx: (j, k)
        # c_idx: (i, k)
        i, j = a_idx
        j_b, k = b_idx
        i_c, k_c = c_idx
        # Note: j must match j_b
        U[i * m + j, col] += coeff
        V[j * p + k, col] += coeff
        W[k * n + i, col] += coeff
        col += 1

    def add_strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        # a_r, a_c: start row, col of A block (2x2)
        # b_r, b_c: start row, col of B block (2x2)
        # c_r, c_c: start row, col of C block (2x2)
        
        # A00: (a_r, a_c), A01: (a_r, a_c+1)
        # A10: (a_r+1, a_c), A11: (a_r+1, a_c+1)
        # Similarly for B and C.
        
        # Strassen formulas for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M7
        
        # We need to add terms for each M.
        # Each M is a scalar product of linear combinations.
        # This requires adding multiple rank-1 terms or using the fact that
        # the rank of 2x2x2 is 7.
        # The 7 terms correspond to the 7 multiplications.
        
        # Let's define the 7 terms explicitly.
        # Term 1: (A00+A11) * (B00+B11)
        # This contributes to C00, C11.
        # We can't just add one term because U, V, W are sparse.
        # We must expand the linear combinations.
        
        # Actually, the standard Strassen decomposition for 2x2x2 uses 7 rank-1 terms.
        # Let's find them.
        # The tensor is the 2x2 multiplication tensor.
        # We can use the known decomposition.
        
        # Here is the decomposition for 2x2x2:
        # r=0: U: A00+A11, V: B00+B11, W: C00+C11? No.
        
        # Let's use the explicit indices.
        # A00 is at (a_r, a_c)
        # A11 is at (a_r+1, a_c+1)
        
        # We will add 7 terms.
        
        # Term 0: (A00+A11)(B00+B11)
        # This is not a single rank-1 term in the global tensor because
        # U would have two non-zeros.
        # Wait, the rank-1 term in the tensor decomposition allows U, V, W to be
        # arbitrary vectors.
        # So U can have non-zeros at (a_r, a_c) and (a_r+1, a_c+1).
        # V can have non-zeros at (b_r, b_c) and (b_r+1, b_c+1).
        # W can have non-zeros at (c_r, c_c) and (c_r+1, c_c+1).
        
        # Let's verify if this works.
        # C = sum r U_r V_r W_r.
        # For r=0:
        # U0 has 1 at A00, 1 at A11.
        # V0 has 1 at B00, 1 at B11.
        # W0 has 1 at C00, 1 at C11?
        # The product will be:
        # (A00+A11)(B00+B11) * (W0 vector).
        # This adds to C00 and C11.
        
        # Let's define the 7 terms for Strassen 2x2.
        
        # 1. M1 = (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11.
        # Contribution to C:
        # C00 += M1
        # C11 += M1
        # So W: C00, C11.
        
        add_strassen_term(
            [(a_r, a_c), (a_r+1, a_c+1)],
            [(b_r, b_c), (b_r+1, b_c+1)],
            [(c_r, c_c), (c_r+1, c_c+1)],
            1.0
        )
        
        # 2. M2 = (A10+A11)B00
        # U: A10, A11. V: B00.
        # Contribution to C:
        # C10 += M2
        # C11 += M2
        # Wait, Strassen formula:
        # C10 = M2 + M4
        # C11 = M1 - M3 + M7
        # My previous mapping of W was wrong.
        
        # Let's use the standard Strassen decomposition for 2x2x2.
        # The 7 terms are:
        # 1. (A00+A11)(B00+B11) -> C00, C11 (with coeffs)
        # This is getting complicated to map manually.
        
        # I will use the schoolbook for 2x2x2 (rank 4) to be safe and simple,
        # then optimize if needed.
        # Rank 4 for 2x2x2.
        # Total rank for 6 blocks: 6 * 4 = 24.
        # Rank 11 for 2x3x2.
        # Total rank: 24 + 6 * 11 = 24 + 66 = 90.
        # This is better than 120 and easier to implement correctly.
        
        # Let's revert to schoolbook for 2x2x2.
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    a_idx = (a_r + di, a_c + dj)
                    b_idx = (b_r + dj, b_c + dk)
                    c_idx = (c_r + di, c_c + dk)
                    add_term(a_idx, b_idx, c_idx, 1.0)

    def add_2x3x2(a_r, a_c, b_r, b_c, c_r, c_c):
        # A: 2x3, B: 3x2, C: 2x2
        # Split B rows into 1 and 2.
        # A cols into 1 and 2.
        
        # Part 1: 2x1x2 (A col 0, B row 0)
        # Rank 4
        for di in range(2):
            for dk in range(2):
                a_idx = (a_r + di, a_c)
                b_idx = (b_r, b_c + dk)
                c_idx = (c_r + di, c_c + dk)
                add_term(a_idx, b_idx, c_idx, 1.0)
                
        # Part 2: 2x2x2 (A cols 1,2, B rows 1,2)
        # Use schoolbook 2x2x2 (rank 4)
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    a_idx = (a_r + di, a_c + 1 + dj)
                    b_idx = (b_r + 1 + dj, b_c + dk)
                    c_idx = (c_r + di, c_c + dk)
                    add_term(a_idx, b_idx, c_idx, 1.0)

    # Call the functions for each block
    
    # 1. A00 B00 -> C00
    add_strassen_2x2(0, 0, 0, 0, 0, 0)
    # 2. A00 B01 -> C01
    add_strassen_2x2(0, 0, 0, 2, 0, 2)
    # 3. A00 B02 -> C02
    add_strassen_2x2(0, 0, 0, 4, 0, 4)
    
    # 4. A10 B00 -> C10
    add_strassen_2x2(2, 0, 0, 0, 2, 0)
    # 5. A10 B01 -> C11
    add_strassen_2x2(2, 0, 0, 2, 2, 2)
    # 6. A10 B02 -> C12
    add_strassen_2x2(2, 0, 0, 4, 2, 4)
    
    # 7. A01 B10 -> C00
    add_2x3x2(0, 2, 2, 0, 0, 0)
    # 8. A01 B11 -> C01
    add_2x3x2(0, 2, 2, 2, 0, 2)
    # 9. A01 B12 -> C02
    add_2x3x2(0, 2, 2, 4, 0, 4)
    
    # 10. A11 B10 -> C10
    add_2x3x2(2, 2, 2, 0, 2, 0)
    # 11. A11 B11 -> C11
    add_2x3x2(2, 2, 2, 2, 2, 2)
    # 12. A11 B12 -> C12
    add_2x3x2(2, 2, 2, 4, 2, 4)
    
    # Check col
    # 6 * 4 + 6 * (4+4) = 24 + 48 = 72?
    # Wait, add_strassen_2x2 uses schoolbook 2x2x2 which is rank 4.
    # 6 * 4 = 24.
    # add_2x3x2 uses rank 4 + rank 4 = 8.
    # 6 * 8 = 48.
    # Total 72.
    
    # My rank estimate was 108 because I thought 2x3x2 was rank 11.
    # But I implemented schoolbook for the 2x2 part inside 2x3x2.
    # Schoolbook 2x2x2 is rank 4.
    # So 2x3x2 is rank 4 (for 2x1x2) + 4 (for 2x2x2) = 8.
    
    # So total rank is 72.
    
    # Let's verify the indices.
    # A00: rows 0,1 cols 0,1.
    # B00: rows 0,1 cols 0,1.
    # C00: rows 0,1 cols 0,1.
    # Correct.
    
    # A01: rows 0,1 cols 2,3,4.
    # B10: rows 2,3,4 cols 0,1.
    # C00: rows 0,1 cols 0,1.
    # Correct.
    
    def to_list(mat):
        return [[round(float(x), 12) for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
