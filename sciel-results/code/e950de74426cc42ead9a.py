import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get block indices
    # Block (bi, bj) contains A[2*bi + r, 2*bj + c] for r,c in 0,1
    def block_indices(bi, bj):
        return [
            (2*bi, 2*bj),
            (2*bi, 2*bj + 1),
            (2*bi + 1, 2*bj),
            (2*bi + 1, 2*bj + 1)
        ]

    # Define outer Strassen terms
    # X_u and Y_u are 2x2 matrices of block references
    # Each entry is (coeff, block_i, block_j) or None
    outer_terms = [
        # u=0: (A00 + A11)(B00 + B11)
        {
            'X': [[(1, 0, 0), None], [None, (1, 1, 1)]],
            'Y': [[(1, 0, 0), None], [None, (1, 1, 1)]]
        },
        # u=1: (A10 + A11)B00
        {
            'X': [[None, None], [(1, 1, 0), (1, 1, 1)]],
            'Y': [[(1, 0, 0), None], [None, None]]
        },
        # u=2: A00(B01 - B11)
        {
            'X': [[(1, 0, 0), None], [None, None]],
            'Y': [[None, (1, 0, 1)], [None, (-1, 1, 1)]]
        },
        # u=3: A11(B10 - B00)
        {
            'X': [[None, None], [None, (1, 1, 1)]],
            'Y': [[(-1, 0, 0), None], [(1, 1, 0), None]]
        },
        # u=4: (A00 + A01)B11
        {
            'X': [[(1, 0, 0), (1, 0, 1)], [None, None]],
            'Y': [[None, None], [None, (1, 1, 1)]]
        },
        # u=5: (A10 - A00)(B00 + B01)
        {
            'X': [[(-1, 0, 0), None], [(1, 1, 0), None]],
            'Y': [[(1, 0, 0), (1, 0, 1)], [None, None]]
        },
        # u=6: (A01 - A11)(B10 + B11)
        {
            'X': [[None, (1, 0, 1)], [None, (-1, 1, 1)]],
            'Y': [[None, None], [(1, 1, 0), (1, 1, 1)]]
        }
    ]

    # Inner Strassen coefficients
    # For C = AB, C_ij = sum coeff * T_k
    # T_0: (a+d)(e+h)
    # T_1: (c+d)e
    # T_2: a(f-h)
    # T_3: d(g-e)
    # T_4: (a+b)h
    # T_5: (c-a)(e+f)
    # T_6: (b-d)(g+h)
    
    # Q coefficients for each T_k (2x2 matrix of coeffs for a,b,c,d)
    # a is top-left, b is top-right, c is bottom-left, d is bottom-right
    Q_coeffs = [
        [[1, 0], [0, 1]], # T_0: a+d
        [[0, 0], [1, 1]], # T_1: c+d
        [[1, 0], [0, 0]], # T_2: a
        [[0, 0], [0, 1]], # T_3: d
        [[1, 1], [0, 0]], # T_4: a+b
        [[-1, 0], [1, 0]],# T_5: c-a
        [[0, 1], [0, -1]] # T_6: b-d
    ]
    
    R_coeffs = [
        [[1, 0], [0, 1]], # T_0: e+h
        [[1, 0], [0, 0]], # T_1: e
        [[0, 1], [0, -1]],# T_2: f-h
        [[0, 0], [1, -1]],# T_3: g-e
        [[0, 0], [0, 1]], # T_4: h
        [[1, 1], [0, 0]], # T_5: e+f
        [[0, 0], [1, 1]]  # T_6: g+h
    ]

    # C reconstruction coefficients for each T_k (2x2 matrix of coeffs for C00, C01, C10, C11)
    # C00 = T0 + T3 - T4 + T6
    # C01 = T2 + T4
    # C10 = T1 + T3
    # C11 = T0 - T1 + T2 + T5
    C_coeffs = [
        [[1, 0], [0, 1]], # T0
        [[0, 0], [1, -1]],# T1
        [[0, 1], [0, 1]], # T2
        [[1, 0], [1, 0]], # T3
        [[-1, 1], [0, 0]],# T4
        [[0, 0], [0, 1]], # T5
        [[1, 0], [0, 0]]  # T6
    ]

    for u in range(7):
        for v in range(7):
            t = u * 7 + v
            
            # Build U and V contributions
            # U corresponds to A, V to B
            # We need to sum contributions from all blocks involved in Q_coeffs[v] and R_coeffs[v]
            
            # Initialize dictionaries for A and B indices
            a_coeffs = {}
            b_coeffs = {}
            
            # Process Q
            for qi in range(2):
                for qj in range(2):
                    coeff_q = Q_coeffs[v][qi][qj]
                    if coeff_q == 0: continue
                    
                    # Get the block reference from outer_terms[u]['X']
                    block_ref = outer_terms[u]['X'][qi][qj]
                    if block_ref is None: continue
                    
                    block_coeff, bi, bj = block_ref
                    total_coeff = coeff_q * block_coeff
                    
                    # Add to a_coeffs
                    indices = block_indices(bi, bj)
                    for idx in indices:
                        if idx not in a_coeffs:
                            a_coeffs[idx] = 0
                        a_coeffs[idx] += total_coeff

            # Process R
            for ri in range(2):
                for rj in range(2):
                    coeff_r = R_coeffs[v][ri][rj]
                    if coeff_r == 0: continue
                    
                    block_ref = outer_terms[u]['Y'][ri][rj]
                    if block_ref is None: continue
                    
                    block_coeff, bi, bj = block_ref
                    total_coeff = coeff_r * block_coeff
                    
                    indices = block_indices(bi, bj)
                    for idx in indices:
                        if idx not in b_coeffs:
                            b_coeffs[idx] = 0
                        b_coeffs[idx] += total_coeff

            # Fill U and V
            for (i, j), coeff in a_coeffs.items():
                if coeff != 0:
                    U[i * m + j][t] = coeff
            
            for (j, k), coeff in b_coeffs.items():
                if coeff != 0:
                    V[j * p + k][t] = coeff

            # Process W (Reconstruction)
            # C_coeffs[v] gives coefficients for the 4 blocks of Z_u
            # Each block of Z_u corresponds to a 2x2 submatrix of C
            for ci in range(2):
                for cj in range(2):
                    coeff_c = C_coeffs[v][ci][cj]
                    if coeff_c == 0: continue
                    
                    # The block Z_u[ci, cj] contributes to C's block (ci, cj)
                    # C's block (ci, cj) contains indices (2*ci + r, 2*cj + s)
                    indices = block_indices(ci, cj)
                    for idx in indices:
                        row, col = idx
                        # W index is k*n + i, where k is col of C, i is row of C?
                        # Prompt: W[k*n + i, r]
                        # C is n x p. C[i, k].
                        # So index is k*n + i.
                        W[col * n + row][t] += coeff_c

    return {"U": U, "V": V, "W": W}
