"""Rank-28 decomposition of 2x2 by 2x8 matrix multiplication using Strassen."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    # Strassen rank-7 decomposition of 2x2x2 multiplication
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B21)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B12+B22)
    # M7 = (A12-A22)(B11+B21)
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M7
    
    # A-side factors for each Strassen term (indexed by [i,j] in 2x2, flattened to i*2+j)
    # Term: (coeff_A11, coeff_A12, coeff_A21, coeff_A22)
    A_factors = [
        (1, 0, 0, 1),   # M1: A11 + A22
        (0, 0, 1, 1),   # M2: A21 + A22
        (1, 0, 0, 0),   # M3: A11
        (0, 0, 0, 1),   # M4: A22
        (1, 1, 0, 0),   # M5: A11 + A12
        (-1, 0, 1, 0),  # M6: A21 - A11
        (0, 1, 0, -1),  # M7: A12 - A22
    ]
    
    # B-side factors for each Strassen term (indexed by [j,k] in 2x2, flattened to j*2+k)
    # Term: (coeff_B11, coeff_B12, coeff_B21, coeff_B22)
    B_factors = [
        (1, 0, 0, 1),   # M1: B11 + B22
        (1, 0, 0, 0),   # M2: B11
        (1, 0, -1, 0),  # M3: B11 - B21
        (0, -1, 0, 1),  # M4: B22 - B12
        (0, 0, 0, 1),   # M5: B22
        (0, 1, 0, 1),   # M6: B12 + B22
        (1, 0, 1, 0),   # M7: B11 + B21
    ]
    
    # C-side factors for each Strassen term (indexed by [i,k] in 2x2, flattened to i*2+k)
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M7
    # Term: (coeff_C11, coeff_C12, coeff_C_C21, coeff_C22)
    C_factors = [
        (1, 0, 0, 1),   # M1: C11 + C22
        (0, 0, 1, 0),   # M2: C21
        (0, 1, 0, -1),  # M3: C12 - C22
        (-1, 0, 1, 0),  # M4: -C11 + C21
        (0, 1, 0, 0),   # M5: C12
        (1, 0, 0, 0),   # M6: C11
        (0, 0, 0, 1),   # M7: C22
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    col = 0
    for q in range(4):  # 4 column pairs
        for s in range(7):  # 7 Strassen terms
            # A-side: U[i*2+j, col]
            a11, a12, a21, a22 = A_factors[s]
            U[0][col] = a11  # i=0, j=0 -> A11
            U[1][col] = a12  # i=0, j=1 -> A12
            U[2][col] = a21  # i=1, j=0 -> A21
            U[3][col] = a22  # i=1, j=1 -> A22
            
            # B-side: V[j*8+k, col], where k = 2*q + r
            b11, b12, b21, b22 = B_factors[s]
            # B11 is at j=0, k=2q (r=0)
            # B12 is at j=0, k=2q+1 (r=1)
            # B21 is at j=1, k=2q (r=0)
            # B22 is at j=1, k=2q+1 (r=1)
            V[0 * 8 + 2 * q + 0][col] = b11
            V[0 * 8 + 2 * q + 1][col] = b12
            V[1 * 8 + 2 * q + 0][col] = b21
            V[1 * 8 + 2 * q + 1][col] = b22
            
            # C-side: W[k*2+i, col], where k = 2*q + r
            c11, c12, c21, c22 = C_factors[s]
            # C11 is at i=0, k=2q (r=0) -> W[(2q)*2 + 0]
            # C12 is at i=0, k=2q+1 (r=1) -> W[(2q+1)*2 + 0]
            # C21 is at i=1, k=2q (r=0) -> W[(2q)*2 + 1]
            # C22 is at i=1, k=2q+1 (r=1) -> W[(2q+1)*2 + 1]
            W[(2 * q + 0) * 2 + 0][col] = c11
            W[(2 * q + 1) * 2 + 0][col] = c12
            W[(2 * q + 0) * 2 + 1][col] = c21
            W[(2 * q + 1) * 2 + 1][col] = c22
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
