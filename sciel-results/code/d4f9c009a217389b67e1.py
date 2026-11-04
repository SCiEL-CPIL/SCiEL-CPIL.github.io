import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # Coefficients for U (A part)
    # Each entry is a tuple of (local_a_idx, coeff)
    # local_a_idx: 0->00, 1->01, 2->10, 3->11
    u_coeffs = [
        [(0, 1), (3, 1)],      # M1: A00 + A11
        [(2, 1), (3, 1)],      # M2: A10 + A11
        [(0, 1)],              # M3: A00
        [(3, 1)],              # M4: A11
        [(0, 1), (1, 1)],      # M5: A00 + A01
        [(2, 1), (0, -1)],     # M6: A10 - A00
        [(1, 1), (3, -1)]      # M7: A01 - A11
    ]
    
    # Coefficients for V (B part)
    # local_b_idx: 0->00, 1->01, 2->10, 3->11
    v_coeffs = [
        [(0, 1), (3, 1)],      # M1: B00 + B11
        [(0, 1)],              # M2: B00
        [(2, 1), (3, -1)],     # M3: B10 - B11
        [(1, 1), (0, -1)],     # M4: B01 - B00
        [(3, 1)],              # M5: B11
        [(0, 1), (1, 1)],      # M6: B00 + B01
        [(2, 1), (3, 1)]       # M7: B10 + B11
    ]
    
    # Coefficients for W (C part)
    # local_c_idx: 0->00, 1->01, 2->10, 3->11
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    w_coeffs = [
        [(0, 1), (3, 1), (2, -1), (1, -1)], # C00
        [(0, 1), (2, 1)],                    # C01
        [(1, 1), (3, 1)],                    # C10
        [(0, 1), (1, -1), (2, 1), (3, 1)]   # C11
    ]
    
    col = 0
    # Part 1: A1 B1
    # A1 uses A cols 0,1. Global j = 0,1.
    # B1 uses B rows 0,1. Global j = 0,1.
    for gamma in range(4):
        for m_idx in range(7):
            # U
            for a_idx, coeff in u_coeffs[m_idx]:
                # a_idx 0->00 (i=0, j'=0), 1->01 (i=0, j'=1), 2->10 (i=1, j'=0), 3->11 (i=1, j'=1)
                i = a_idx // 2
                jp = a_idx % 2
                j = jp # Global j for A1
                U[i * m + j][col] += coeff
                
            # V
            for b_idx, coeff in v_coeffs[m_idx]:
                # b_idx 0->00 (j'=0, k'=0), 1->01 (j'=0, k'=1), 2->10 (j'=1, k'=0), 3->11 (j'=1, k'=1)
                jp = b_idx // 2
                kp = b_idx % 2
                j = jp # Global j for B1
                k = 2 * gamma + kp # Global k
                V[j * p + k][col] += coeff
                
            # W
            for c_idx, coeff in w_coeffs[m_idx]:
                # c_idx 0->00 (i=0, k'=0), 1->01 (i=0, k'=1), 2->10 (i=1, k'=0), 3->11 (i=1, k'=1)
                i = c_idx // 2
                kp = c_idx % 2
                k = 2 * gamma + kp # Global k
                W[k * n + i][col] += coeff
                
            col += 1
            
    # Part 2: A2 B2
    # A2 uses A cols 2,3. Global j = 2,3.
    # B2 uses B rows 2,3. Global j = 2,3.
    for gamma in range(4):
        for m_idx in range(7):
            # U
            for a_idx, coeff in u_coeffs[m_idx]:
                i = a_idx // 2
                jp = a_idx % 2
                j = 2 + jp # Global j for A2
                U[i * m + j][col] += coeff
                
            # V
            for b_idx, coeff in v_coeffs[m_idx]:
                jp = b_idx // 2
                kp = b_idx % 2
                j = 2 + jp # Global j for B2
                k = 2 * gamma + kp # Global k
                V[j * p + k][col] += coeff
                
            # W
            for c_idx, coeff in w_coeffs[m_idx]:
                i = c_idx // 2
                kp = c_idx % 2
                k = 2 * gamma + kp # Global k
                W[k * n + i][col] += coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
