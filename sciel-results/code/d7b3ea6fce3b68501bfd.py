import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using 3x3 blocks of 2x2 matrices
    # Each 2x2 multiplication uses Strassen's rank 7 algorithm
    # Total rank = 3*3*3 (block positions) * 7 (Strassen rank) = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen rank 7 coefficients for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # U2[r] corresponds to A coefficients for term r
    # V2[r] corresponds to B coefficients for term r
    # W2[r] corresponds to C coefficients for term r
    # Indices: 0=a, 1=b, 2=c, 3=d for A/B
    # Indices: 0=C11, 1=C12, 2=C21, 3=C22 for C
    
    U2 = [
        [1, 0, 0, 1], # m1: a+d
        [0, 0, 1, 1], # m2: c+d
        [1, 0, 0, 0], # m3: a
        [0, 0, 0, 1], # m4: d
        [1, 1, 0, 0], # m5: a+b
        [0, 0, 1, -1], # m6: c-a
        [0, 1, 0, -1]  # m7: b-d
    ]
    
    V2 = [
        [1, 1, 0, 0], # m1: e+h
        [1, 0, 0, 0], # m2: e
        [0, 1, 0, -1], # m3: f-h
        [0, -1, 1, 0], # m4: g-e
        [0, 0, 0, 1], # m5: h
        [1, 1, 0, 0], # m6: e+f
        [0, 0, 1, 1]  # m7: g+h
    ]
    
    W2 = [
        [1, 0, 0, 1], # m1: C11 + C22
        [0, 0, 1, -1], # m2: C21 - C22
        [0, 1, 0, 1], # m3: C12 + C22
        [1, 0, 1, 0], # m4: C11 + C21
        [-1, 1, 0, 0], # m5: -C11 + C12
        [0, 0, 0, 1], # m6: C22
        [1, 0, 0, 0]  # m7: C11
    ]
    
    # Note: The W2 coefficients must satisfy C = sum W2[r] * m_r
    # Let's verify m1: C11 += 1, C22 += 1. Correct.
    # m2: C21 += 1, C22 -= 1. Correct (C22 = ... - m2 ...).
    # m3: C12 += 1, C22 += 1. Correct (C22 = ... + m3 ...).
    # m4: C11 += 1, C21 += 1. Correct.
    # m5: C11 -= 1, C12 += 1. Correct.
    # m6: C22 += 1. Correct.
    # m7: C11 += 1. Correct.
    
    # Iterate over 3x3 block positions
    # i: block row of C (and A)
    # j: block col of C (and B)
    # k: summation index (col of A, row of B)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for r in range(7):
                    # Calculate column index
                    # Order: (i, j, k, r)
                    # i,j,k in 0..2, r in 0..6
                    col = ((i * 3 + j) * 3 + k) * 7 + r
                    
                    # Fill U
                    # U corresponds to A block (i, k)
                    # Local indices 0..3 map to global indices in block (i, k)
                    for x in range(2):
                        for y in range(2):
                            local_idx = x * 2 + y
                            global_row = i * 2 + x
                            global_col = k * 2 + y
                            flat_idx = global_row * 6 + global_col
                            U[flat_idx][col] = U2[r][local_idx]
                            
                    # Fill V
                    # V corresponds to B block (k, j)
                    for x in range(2):
                        for y in range(2):
                            local_idx = x * 2 + y
                            global_row = k * 2 + x
                            global_col = j * 2 + y
                            flat_idx = global_row * 6 + global_col
                            V[flat_idx][col] = V2[r][local_idx]
                            
                    # Fill W
                    # W corresponds to C block (i, j)
                    for x in range(2):
                        for y in range(2):
                            local_idx = x * 2 + y
                            global_row = i * 2 + x
                            global_col = j * 2 + y
                            flat_idx = global_row * 6 + global_col
                            W[flat_idx][col] = W2[r][local_idx]
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
