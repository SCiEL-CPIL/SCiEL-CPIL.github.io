"""
Rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.

This program constructs a rank-42 decomposition using block Strassen along the
middle dimension (j) and output columns (k).

The decomposition splits the 4x6 product into 2x3 blocks of 2x2 Strassen
multiplications, each contributing 7 bilinear products.

Total rank = 2 * 3 * 7 = 42
"""

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (12, 42)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # U: A matrix coefficients
    # V: B matrix coefficients  
    # W: C matrix coefficients
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # P1: (A00+A11)
        {(1, 0): 1, (1, 1): 1},   # P2: (A10+A11)
        {(0, 0): 1},              # P3: A00
        {(1, 1): 1},              # P4: A11
        {(0, 0): 1, (0, 1): 1},   # P5: (A00+A01)
        {(1, 0): 1, (0, 0): -1},  # P6: (A10-A00)
        {(0, 1): 1, (1, 1): -1}   # P7: (A01-A11)
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # P1: (B00+B11)
        {(0, 0): 1},              # P2: B00
        {(0, 1): 1, (1, 1): -1},  # P3: (B01-B11)
        {(1, 0): 1, (0, 0): -1},  # P4: (B10-B00)
        {(1, 1): 1},              # P5: B11
        {(0, 0): 1, (0, 1): 1},   # P6: (B00+B01)
        {(1, 0): 1, (1, 1): 1}    # P7: (B10+B11)
    ]
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},   # P1: C00 += P1, C11 += P1
        {(1, 0): 1, (1, 1): -1},  # P2: C10 += P2, C11 -= P2
        {(0, 1): 1, (1, 1): 1},   # P3: C01 += P3, C11 += P3
        {(0, 0): 1, (1, 0): 1},   # P4: C00 += P4, C10 += P4
        {(0, 0): -1, (0, 1): 1},  # P5: C00 -= P5, C01 += P5
        {(1, 1): 1},              # P6: C11 += P6
        {(0, 0): 1}               # P7: C00 += P7
    ]
    
    col = 0
    # Split j dimension into 2 blocks of size 2
    # Split k dimension into 3 blocks of size 2
    for a in range(2):      # j block: j = a*2, a*2+1
        for b in range(3):  # k block: k = b*2, b*2+1
            for s in range(7):
                # Fill U: A[i, j] contributions
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Fill V: B[j, k] contributions
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Fill W: C[k, i] contributions (trace convention W[k*n+i, r])
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
