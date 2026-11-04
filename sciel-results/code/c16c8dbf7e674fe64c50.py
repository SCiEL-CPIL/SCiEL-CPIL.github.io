"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using two instances of Laderman's rank-19 algorithm for 4x4 multiplication
    rank = 38
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 19 products for Laderman's algorithm
    # Each product is a tuple of (A_indices, B_indices, C_indices, coeff)
    # A_indices: list of (i, j, coeff)
    # B_indices: list of (j, k, coeff)
    # C_indices: list of (k, i, coeff)
    
    laderman_terms = []
    
    # Term 1: (A00 + A11)(B00 + B11)
    laderman_terms.append((
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        [(0,0,1), (0,1,1), (1,0,1), (1,1,1)] # C00, C01, C10, C11
    ))
    # Term 2: (A00 - A11)(B10 - B01)
    laderman_terms.append((
        [(0,0,1), (1,1,-1)],
        [(1,0,1), (0,1,-1)],
        [(0,0,-1), (0,1,1), (1,0,1), (1,1,-1)] # C00, C01, C10, C11
    ))
    # Term 3: (A01 + A10)(B00 + B11)
    laderman_terms.append((
        [(0,1,1), (1,0,1)],
        [(0,0,1), (1,1,1)],
        [(0,0,1), (0,1,-1), (1,0,1), (1,1,-1)] # C00, C01, C10, C11
    ))
    # Term 4: (A01 - A10)(B01 + B10)
    laderman_terms.append((
        [(0,1,1), (1,0,-1)],
        [(0,1,1), (1,0,1)],
        [(0,0,-1), (0,1,1), (1,0,-1), (1,1,1)] # C00, C01, C10, C11
    ))
    
    # Terms 5-8 involve A02, A03, A12, A13 and B20, B30, B23, B32
    # These contribute to C02, C03, C12, C13
    # Term 5: (A02 + A13)(B20 + B30)
    laderman_terms.append((
        [(0,2,1), (1,3,1)],
        [(2,0,1), (3,0,1)],
        [(2,0,1), (2,1,1), (3,0,1), (3,1,1)] # C20, C21, C30, C31
    ))
    # Term 6: (A02 - A13)(B30 - B20)
    laderman_terms.append((
        [(0,2,1), (1,3,-1)],
        [(3,0,1), (2,0,-1)],
        [(2,0,-1), (2,1,1), (3,0,1), (3,1,-1)] # C20, C21, C30, C31
    ))
    # Term 7: (A03 + A12)(B20 + B30)
    laderman_terms.append((
        [(0,3,1), (1,2,1)],
        [(2,0,1), (3,0,1)],
        [(2,0,1), (2,1,-1), (3,0,1), (3,1,-1)] # C20, C21, C30, C31
    ))
    # Term 8: (A03 - A12)(B23 + B32)
    laderman_terms.append((
        [(0,3,1), (1,2,-1)],
        [(2,3,1), (3,2,1)],
        [(2,0,-1), (2,1,1), (3,0,-1), (3,1,1)] # C20, C21, C30, C31
    ))
    
    # Terms 9-12 involve A20, A21, A30, A31 and B00, B01, B10, B11
    # These contribute to C20, C21, C30, C31
    # Term 9: (A20 + A31)(B00 + B11)
    laderman_terms.append((
        [(2,0,1), (3,1,1)],
        [(0,0,1), (1,1,1)],
        [(0,2,1), (0,3,1), (1,2,1), (1,3,1)] # C02, C03, C12, C13
    ))
    # Term 10: (A20 - A31)(B10 - B01)
    laderman_terms.append((
        [(2,0,1), (3,1,-1)],
        [(1,0,1), (0,1,-1)],
        [(0,2,-1), (0,3,1), (1,2,1), (1,3,-1)] # C02, C03, C12, C13
    ))
    # Term 11: (A21 + A30)(B00 + B11)
    laderman_terms.append((
        [(2,1,1), (3,0,1)],
        [(0,0,1), (1,1,1)],
        [(0,2,1), (0,3,-1), (1,2,1), (1,3,-1)] # C02, C03, C12, C13
    ))
    # Term 12: (A21 - A30)(B01 + B10)
    laderman_terms.append((
        [(2,1,1), (3,0,-1)],
        [(0,1,1), (1,0,1)],
        [(0,2,-1), (0,3,1), (1,2,-1), (1,3,1)] # C02, C03, C12, C13
    ))
    
    # Terms 13-16 involve A22, A23, A32, A33 and B20, B30, B23, B32
    # These contribute to C22, C23, C32, C33
    # Term 13: (A22 + A33)(B20 + B30)
    laderman_terms.append((
        [(2,2,1), (3,3,1)],
        [(2,0,1), (3,0,1)],
        [(2,2,1), (2,3,1), (3,2,1), (3,3,1)] # C22, C23, C32, C33
    ))
    # Term 14: (A22 - A33)(B30 - B20)
    laderman_terms.append((
        [(2,2,1), (3,3,-1)],
        [(3,0,1), (2,0,-1)],
        [(2,2,-1), (2,3,1), (3,2,1), (3,3,-1)] # C22, C23, C32, C33
    ))
    # Term 15: (A23 + A32)(B20 + B30)
    laderman_terms.append((
        [(2,3,1), (3,2,1)],
        [(2,0,1), (3,0,1)],
        [(2,2,1), (2,3,-1), (3,2,1), (3,3,-1)] # C22, C23, C32, C33
    ))
    # Term 16: (A23 - A32)(B23 + B32)
    laderman_terms.append((
        [(2,3,1), (3,2,-1)],
        [(2,3,1), (3,2,1)],
        [(2,2,-1), (2,3,1), (3,2,-1), (3,3,1)] # C22, C23, C32, C33
    ))
    
    # Terms 17-19 involve cross terms
    # Term 17: (A00 + A22)(B00 + B22)
    laderman_terms.append((
        [(0,0,1), (2,2,1)],
        [(0,0,1), (2,2,1)],
        [(0,0,1), (0,2,1), (2,0,1), (2,2,1)] # C00, C02, C20, C22
    ))
    # Term 18: (A00 - A22)(B20 - B02)
    laderman_terms.append((
        [(0,0,1), (2,2,-1)],
        [(2,0,1), (0,2,-1)],
        [(0,0,-1), (0,2,1), (2,0,1), (2,2,-1)] # C00, C02, C20, C22
    ))
    # Term 19: (A02 + A20)(B00 + B22)
    laderman_terms.append((
        [(0,2,1), (2,0,1)],
        [(0,0,1), (2,2,1)],
        [(0,0,1), (0,2,-1), (2,0,1), (2,2,-1)] # C00, C02, C20, C22
    ))
    
    # Fill U, V, W for the first 19 terms
    for t in range(19):
        a_terms, b_terms, c_terms = laderman_terms[t]
        for (i, j, coeff) in a_terms:
            U[i*m + j][t] += coeff
        for (j, k, coeff) in b_terms:
            V[j*p + k][t] += coeff
        for (k, i, coeff) in c_terms:
            W[k*n + i][t] += coeff
            
    # Fill U, V, W for the next 19 terms (second 4x4 block of B)
    for t in range(19):
        a_terms, b_terms, c_terms = laderman_terms[t]
        col = 19 + t
        for (i, j, coeff) in a_terms:
            U[i*m + j][col] += coeff
        for (j, k, coeff) in b_terms:
            # B index is j*p + (4+k)
            V[j*p + (4 + k)][col] += coeff
        for (k, i, coeff) in c_terms:
            # C index is (4+k)*n + i
            W[(4 + k)*n + i][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
