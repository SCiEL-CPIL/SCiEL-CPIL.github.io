"""Rewritten program for 2x4x8 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Split the 2x4x8 multiplication into 4 independent 2x4x2 sub-problems.
    # For each q in 0..3, multiply a 2x4 block of A by a 4x2 block of B.
    # Each 2x4x2 multiplication is decomposed into two 2x2x2 multiplications
    # (for columns 0-1 and 2-3 of the 4x2 block), each using Strassen's
    # rank-7 algorithm. Total rank: 4 * 2 * 7 = 56.
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen's 7 multiplication factors for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen term c (0..6), define:
    # a_coeffs: list of (i, j, coeff) for the 2x2 A matrix
    # b_coeffs: list of (j, k, coeff) for the 2x2 B matrix
    # c_coeffs: list of (i, k, coeff) for the 2x2 C matrix
    
    strassen_terms = [
        # c=0: M1 = (A00+A11)(B00+B11)
        {
            'a': [(0,0,1), (1,1,1)],
            'b': [(0,0,1), (1,1,1)],
            'c': [(0,0,1), (1,1,-1)]  # M1 appears in C00 with +1, C11 with -1
        },
        # c=1: M2 = (A10+A11)B00
        {
            'a': [(1,0,1), (1,1,1)],
            'b': [(0,0,1)],
            'c': [(1,0,1), (1,1,-1)]  # M2 appears in C10 with +1, C11 with -1
        },
        # c=2: M3 = A00(B01-B11)
        {
            'a': [(0,0,1)],
            'b': [(0,1,1), (1,1,-1)],
            'c': [(0,0,-1), (0,1,1)]  # M3 appears in C00 with -1, C01 with +1
        },
        # c=3: M4 = A11(B10-B00)
        {
            'a': [(1,1,1)],
            'b': [(1,0,1), (0,0,-1)],
            'c': [(0,0,1), (1,0,1)]  # M4 appears in C00 with +1, C10 with +1
        },
        # c=4: M5 = (A00+A01)B11
        {
            'a': [(0,0,1), (0,1,1)],
            'b': [(1,1,1)],
            'c': [(0,0,-1), (0,1,1)]  # M5 appears in C00 with -1, C01 with +1
        },
        # c=5: M6 = (A10-A00)(B00+B01)
        {
            'a': [(1,0,1), (0,0,-1)],
            'b': [(0,0,1), (0,1,1)],
            'c': [(1,1,1)]  # M6 appears in C11 with +1
        },
        # c=6: M7 = (A01-A11)(B10+B11)
        {
            'a': [(0,1,1), (1,1,-1)],
            'b': [(1,0,1), (1,1,1)],
            'c': [(0,0,1), (1,1,1)]  # M7 appears in C00 with +1, C11 with +1
        },
    ]
    
    for q in range(4):
        for sub in range(2):
            # sub=0: columns 2q, 2q+1 of B (i.e., j=0,1 in the 2x2 B block)
            # sub=1: columns 2q+2, 2q+3 of B (i.e., j=2,3 in the 2x2 B block)
            col_base = q * 14 + sub * 7
            for c in range(7):
                col = col_base + c
                term = strassen_terms[c]
                
                # Set U entries
                for (i_local, j_local, coeff) in term['a']:
                    # Global A index: row i_local (0 or 1), col j_local + sub*2
                    # In the 2x4 A matrix, columns are 0,1,2,3
                    # For sub=0: j_global = j_local + 0*2 = j_local
                    # For sub=1: j_global = j_local + 1*2 = j_local + 2
                    j_global = j_local + sub * 2
                    u_idx = i_local * m + j_global
                    U[u_idx][col] = coeff
                
                # Set V entries
                for (j_local, k_local, coeff) in term['b']:
                    # Global B index: row j_local + sub*2 (within the 4-row block), col k_local + 2*q
                    # In the 4x8 B matrix, rows are 0,1,2,3 and cols are 0..7
                    # For sub=0: row_global = j_local + 0 = j_local
                    # For sub=1: row_global = j_local + 2
                    # col_global = k_local + 2*q
                    j_global = j_local + sub * 2
                    k_global = k_local + 2 * q
                    v_idx = j_global * p + k_global
                    V[v_idx][col] = coeff
                
                # Set W entries
                for (i_local, k_local, coeff) in term['c']:
                    # Global C index: row i_local, col 2*q + k_local
                    # W index: (k_global * n + i_global)
                    k_global = k_local + 2 * q
                    w_idx = k_global * n + i_local
                    W[w_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
