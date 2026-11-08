import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # rank = 33
    rank = 33
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Part 1: A_1 B_1
    # A_1 is col 0 (j=0), B_1 is row 0 (j=0)
    # A_{i,0} B_{0,k} -> C_{i,k}
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Part 2: A_2 B_2
    # A_2 is cols 1,2 (j=1,2), B_2 is rows 1,2 (j=1,2)
    # Split B_2 into 3 blocks of 2 columns each.
    # Block m_idx corresponds to columns 2*m_idx, 2*m_idx+1
    for m_idx in range(3):
        # Local indices i' in {0,1}, j' in {0,1}, k' in {0,1}
        # Global i = i', j = 1 + j', k = 2*m_idx + k'
        
        # Helper to add Strassen term
        # term: (sum of A coeffs) * (sum of B coeffs) * (sum of C coeffs)
        # A coeffs: list of (i', j', coeff)
        # B coeffs: list of (j', k', coeff)
        # C coeffs: list of (i', k', coeff)
        
        def add_strassen_term(a_coeffs, b_coeffs, c_coeffs):
            nonlocal col
            for i_p, j_p, ca in a_coeffs:
                U[i_p * m + (1 + j_p), col] += ca
            for j_p, k_p, cb in b_coeffs:
                V[(1 + j_p) * p + (2 * m_idx + k_p), col] += cb
            for i_p, k_p, cc in c_coeffs:
                W[(2 * m_idx + k_p) * n + i_p, col] += cc
            col += 1
            
        # M1 = (A00 + A11)(B00 + B11)
        # C00 += M1, C11 += M1
        add_strassen_term(
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)]
        )
        
        # M2 = (A10 + A11)B00
        # C10 += M2
        add_strassen_term(
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(1, 0, 1)]
        )
        
        # M3 = A00(B01 - B11)
        # C01 += M3
        add_strassen_term(
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(0, 1, 1)]
        )
        
        # M4 = A11(B10 - B00)
        # C00 += M4, C10 += M4
        add_strassen_term(
            [(1, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (1, 0, 1)]
        )
        
        # M5 = (A00 + A01)B11
        # C01 += M5, C00 -= M5
        add_strassen_term(
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)],
            [(0, 1, 1), (0, 0, -1)]
        )
        
        # M6 = (A10 - A00)(B00 + B01)
        # C11 += M6
        add_strassen_term(
            [(1, 0, 1), (0, 0, -1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)]
        )
        
        # M7 = (A01 - A11)(B10 + B11)
        # C00 += M7
        add_strassen_term(
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)]
        )
        
    return {"U": U, "V": V, "W": W}
