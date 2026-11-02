import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)b00
    # M2 = a00(b01-b11)
    # M3 = a11(b10-b00)
    # M4 = (a00+a01)b11
    # M5 = (a10-a00)(b00+b01)
    # M6 = (a01-a11)(b10+b11)
    
    # Indices for A (2x3): row i, col j. U[i*m+j]
    # Indices for B (3x10): row j, col k. V[j*p+k]
    # Indices for C (2x10): row i, col k. W[k*n+i]
    
    # Strassen factor definitions
    # a_idx, b_idx, w_coeffs for c00, c01, c10, c11
    # a_idx is tuple of (i,j, coeff)
    # b_idx is tuple of (j,k, coeff)
    # w_coeffs is dict for local c indices (0,0), (0,1), (1,0), (1,1)
    
    strassen_factors = [
        {
            'a': [(0,0,1), (1,1,1)],
            'b': [(0,0,1), (1,1,1)],
            'w': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}
        },
        {
            'a': [(1,0,1), (1,1,1)],
            'b': [(0,0,1)],
            'w': {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}
        },
        {
            'a': [(0,0,1)],
            'b': [(0,1,1), (1,1,-1)],
            'w': {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}
        },
        {
            'a': [(1,1,1)],
            'b': [(1,0,1), (0,0,-1)],
            'w': {(0,0): 1, (0,1): 0, (1,0): -1, (1,1): 0}
        },
        {
            'a': [(0,0,1), (0,1,1)],
            'b': [(1,1,1)],
            'w': {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}
        },
        {
            'a': [(1,0,1), (0,0,-1)],
            'b': [(0,0,1), (0,1,1)],
            'w': {(0,0): -1, (0,1): 1, (1,0): 1, (1,1): 0}
        },
        {
            'a': [(0,1,1), (1,1,-1)],
            'b': [(1,0,1), (1,1,1)],
            'w': {(0,0): 1, (0,1): -1, (1,0): 0, (1,1): 0}
        }
    ]
    
    # Outer product factors for a1 * b1
    # a1 is col 2 of A (index 2). b1 is row 2 of B (index 2).
    # C = a1 b1.
    # c00 = a02 b20
    # c01 = a02 b21
    # c10 = a12 b20
    # c11 = a12 b21
    outer_factors = [
        {
            'a': [(0,2,1)],
            'b': [(2,0,1)],
            'w': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}
        },
        {
            'a': [(0,2,1)],
            'b': [(2,1,1)],
            'w': {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}
        },
        {
            'a': [(1,2,1)],
            'b': [(2,0,1)],
            'w': {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}
        },
        {
            'a': [(1,2,1)],
            'b': [(2,1,1)],
            'w': {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}
        }
    ]
    
    # Build U, V, W
    for block in range(5):
        # Global column offset for B and C
        k_offset = block * 2
        
        for s_idx in range(11):
            col = block * 11 + s_idx
            
            if s_idx < 7:
                factor = strassen_factors[s_idx]
            else:
                factor = outer_factors[s_idx - 7]
            
            # Fill U
            for (i, j, coeff) in factor['a']:
                U[i * m + j][col] += coeff
                
            # Fill V
            for (j, k_local, coeff) in factor['b']:
                k_global = k_offset + k_local
                V[j * p + k_global][col] += coeff
                
            # Fill W
            for ((i_local, k_local), coeff) in factor['w'].items():
                k_global = k_offset + k_local
                # W index: k*n + i
                W[k_global * n + i_local][col] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
