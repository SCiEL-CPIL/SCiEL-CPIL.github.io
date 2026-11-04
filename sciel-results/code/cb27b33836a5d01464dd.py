"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Decomposition of M_{3,5,5} using Strassen for 2x2 blocks
    # Split A (3x5) into A2 (3x2) and A1 (3x1)
    # Split B (5x5) into B2 (2x5) and B1 (1x5)
    # C = A2 B2 + A1 B1
    
    # A1 B1 is M_{3,1,5}. Rank 15.
    # A2 B2 is M_{3,2,5}.
    # Split B2 (2x5) into B22 (2x2), B22' (2x2), B21 (2x1)
    # C22 = A2 B22 is M_{3,2,2}. Rank 11.
    # C22' = A2 B22' is M_{3,2,2}. Rank 11.
    # C21 = A2 B21 is M_{3,2,1}. Rank 6.
    
    # Total Rank = 15 + 11 + 11 + 6 = 43.
    
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Part 1: A1 B1 (M_{3,1,5})
    # A1 is column 2 of A (index 2 in m).
    # B1 is row 4 of B (index 4 in m).
    # C_{i,k} = A_{i,2} B_{4,k}
    for i in range(3):
        for k in range(5):
            add_term(i * 5 + 2, 4 * 5 + k, k * 3 + i, 1, 1, 1)
            
    # Part 2: A2 B2 (M_{3,2,5})
    # A2 is columns 0,1 of A.
    # B2 is rows 0,1 of B.
    # Split B2 into B22 (cols 0,1), B22' (cols 2,3), B21 (col 4)
    
    # Subpart 2.1: A2 B22 (M_{3,2,2}) -> C_{i,0:2}
    # Use Strassen for rows 0,1 of A2 and standard for row 2.
    # Strassen 7 terms for A2[0:2, :] and B22[:, 0:2]
    # A2 indices: i in 0..2, j in 0..1.
    # B22 indices: j in 0..1, k in 0..1.
    
    # Strassen products for M_{2,2,2} part (rows 0,1 of A)
    # M1 = (A00+A11)(B00+B11)
    add_term(0*5+0, 0*5+0, 0*3+0, 1, 1, 1) # A00 B00
    add_term(0*5+0, 1*5+1, 0*3+0, 1, 1, 1) # A00 B11
    add_term(1*5+1, 0*5+0, 0*3+0, 1, 1, 1) # A11 B00
    add_term(1*5+1, 1*5+1, 0*3+0, 1, 1, 1) # A11 B11
    # This is getting complex to inline. 
    # I will use a helper to add Strassen 7 terms.
    
    def add_strassen_2x2(a_row_offset, b_col_offset, c_col_offset, a_col_base, b_row_base):
        # a_row_offset: start row in A (0 or 1 for the 2x2 block)
        # b_col_offset: start col in B (0 or 1 for the 2x2 block)
        # c_col_offset: start col in C (0 or 1 for the 2x2 block)
        # a_col_base: column index in A for the 2x2 block (0 for A2)
        # b_row_base: row index in B for the 2x2 block (0 for B2)
        
        # Indices in U, V, W
        # A is 3x5. A[i, a_col_base + j]
        # B is 5x5. B[b_row_base + j, b_col_offset + k]
        # C is 3x5. C[i, c_col_offset + k]
        
        # 7 Terms
        # 1: (A00+A11)(B00+B11)
        add_term((a_row_offset+0)*5 + (a_col_base+0), (b_row_base+0)*5 + (b_col_offset+0), (b_col_offset+0)*3 + (a_row_offset+0), 1, 1, 1)
        add_term((a_row_offset+0)*5 + (a_col_base+0), (b_row_base+1)*5 + (b_col_offset+1), (b_col_offset+0)*3 + (a_row_offset+0), 1, 1, 1)
        add_term((a_row_offset+1)*5 + (a_col_base+1), (b_row_base+0)*5 + (b_col_offset+0), (b_col_offset+0)*3 + (a_row_offset+0), 1, 1, 1)
        add_term((a_row_offset+1)*5 + (a_col_base+1), (b_row_base+1)*5 + (b_col_offset+1), (b_col_offset+0)*3 + (a_row_offset+0), 1, 1, 1)
        
        # This is incorrect. Strassen mixes inputs.
        # I need to define the linear forms.
        
    # Due to complexity, I will stick to the 75 rank if I cannot safely write 43.
    # However, 43 is a valid improvement.
    # I will implement the 43 rank carefully.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Helper to add a simple term
    def add_simple(ai, aj, bj, bk, ci, ck):
        nonlocal col
        U[ai * 5 + aj][col] = 1
        V[bj * 5 + bk][col] = 1
        W[bk * 3 + ci][col] = 1
        col += 1
        
    # Helper to add a weighted term
    def add_weighted(ai, aj, bj, bk, ci, ck, val):
        nonlocal col
        U[ai * 5 + aj][col] = val
        V[bj * 5 + bk][col] = val
        W[bk * 3 + ci][col] = val
        col += 1
        
    # Part 1: A1 B1 (M_{3,1,5}) - 15 terms
    for i in range(3):
        for k in range(5):
            add_simple(i, 2, 4, k, i, k)
            
    # Part 2: A2 B2 (M_{3,2,5})
    # B2 cols 0,1,2,3,4
    # B22: cols 0,1
    # B22': cols 2,3
    # B21: col 4
    
    # Subpart 2.1: A2 B22 (M_{3,2,2}) - 11 terms
    # Strassen for rows 0,1 of A2 (cols 0,1) and B22 (rows 0,1, cols 0,1)
    # A2 rows 0,1 are A[0,0:2], A[1,0:2]
    # B22 is B[0:2, 0:2]
    
    # 7 Strassen terms
    # M1 = (A00+A11)(B00+B11)
    # We need to add terms that sum to M1.
    # M1 = A00 B00 + A00 B11 + A11 B00 + A11 B11
    add_simple(0, 0, 0, 0, 0, 0)
    add_simple(0, 0, 1, 1, 0, 0)
    add_simple(1, 1, 0, 0, 0, 0)
    add_simple(1, 1, 1, 1, 0, 0)
    
    # M2 = (A10+A11) B00
    add_simple(0, 0, 0, 0, 1, 0)
    add_simple(1, 1, 0, 0, 1, 0)
    
    # M3 = A00 (B10 - B11)
    add_simple(0, 0, 1, 0, 0, 1)
    add_weighted(0, 0, 1, 1, 0, 1, -1)
    
    # M4 = A11 (B01 - B00)
    add_simple(1, 1, 0, 1, 0, 1)
    add_weighted(1, 1, 0, 0, 0, 1, -1)
    
    # M5 = (A00+A01) B11
    add_simple(0, 0, 1, 1, 0, 1)
    add_simple(0, 1, 1, 1, 0, 1)
    
    # M6 = (A10 - A00) (B00 + B01)
    add_simple(0, 0, 0, 0, 1, 0)
    add_simple(0, 0, 0, 1, 1, 0)
    add_weighted(0, 0, 0, 0, 1, 0, -1)
    add_weighted(0, 0, 0, 1, 1, 0, -1)
    
    # M7 = (A01 - A11) (B10 + B11)
    add_simple(0, 1, 1, 0, 0, 1)
    add_simple(0, 1, 1, 1, 0, 1)
    add_weighted(1, 1, 1, 0, 0, 1, -1)
    add_weighted(1, 1, 1, 1, 0, 1, -1)
    
    # Now we have 7+4+2+2+2+4+4 = 25 terms?
    # Strassen is 7 products. My expansion is 25 simple terms.
    # This is not reducing rank.
    # I must define the 7 products as single terms in the decomposition.
    
    # I will return the schoolbook decomposition to ensure correctness.
    U = [[0] * 75 for _ in range(n * m)]
    V = [[0] * 75 for _ in range(m * p)]
    W = [[0] * 75 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
