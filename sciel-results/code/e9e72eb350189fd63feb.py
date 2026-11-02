"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35: 7 Strassen terms * 5 components per term
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # b00, b01, b10, b11 are vectors of length 5
    
    # Term definitions: (coeffs for A, coeffs for B blocks)
    # B blocks: 0=b00, 1=b01, 2=b10, 3=b11
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    
    terms = [
        # M1: (a00+a11)(b00+b11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2: (a10+a11)b00
        ([(2, 1), (3, 1)], [(0, 1)]),
        # M3: a00(b01-b11)
        ([(0, 1)], [(1, 1), (3, -1)]),
        # M4: a11(b10-b00)
        ([(3, 1)], [(2, 1), (0, -1)]),
        # M5: (a00+a01)b11
        ([(0, 1), (1, 1)], [(3, 1)]),
        # M6: (a10-a00)(b00+b01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # M7: (a01-a11)(b10+b11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
    ]
    
    # Reconstruction coefficients for C blocks
    # C00 (row 0, cols 0-4), C01 (row 0, cols 5-9)
    # C10 (row 1, cols 0-4), C11 (row 1, cols 5-9)
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    recon = [
        [(0, 1), (3, -1), (4, -1), (6, 1)],   # C00 = M1 - M4 - M5 + M7
        [(2, 1), (4, 1)],                     # C01 = M3 + M5
        [(1, 1), (3, 1)],                     # C10 = M2 + M4
        [(0, 1), (2, -1), (5, -1), (6, 1)],   # C11 = M1 - M3 - M6 + M7
    ]
    
    # Fill U, V, W
    for t_idx, (a_coeffs, b_coeffs) in enumerate(terms):
        for comp in range(5):
            r = t_idx * 5 + comp
            
            # U: indices i*m+j
            for (a_idx, coeff) in a_coeffs:
                i, j = divmod(a_idx, m)
                U[i * m + j][r] = coeff
            
            # V: indices j*p+k
            # b_coeffs are for blocks b00, b01, b10, b11
            # b00: j=0, k=0..4
            # b01: j=0, k=5..9
            # b10: j=1, k=0..4
            # b11: j=1, k=5..9
            for (b_block, coeff) in b_coeffs:
                j = b_block // 2
                k_start = (b_block % 2) * 5
                for k in range(k_start, k_start + 5):
                    V[j * p + k][r] = coeff
            
            # W: indices k*n+i
            for (c_block, coeff) in recon[t_idx]:
                i = c_block // 2
                k_start = (c_block % 2) * 5
                for k in range(k_start, k_start + 5):
                    W[k * n + i][r] = coeff
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
