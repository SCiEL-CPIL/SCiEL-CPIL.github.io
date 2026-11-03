"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    # Use Strassen-based block decomposition for rank 42
    # A (2x4) split into A0, A1 (each 2x2)
    # B (4x6) split into 3 column-blocks of 4x2, each further split into top/bottom 2x2
    # C (2x6) = A0 * B_top + A1 * B_bot
    # Each (2x2)*(2x6) = 3 Strassen (2x2)*(2x2) products
    # Total: 6 Strassen products * rank 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # (8, 42)
    V = [[0] * rank for _ in range(m * p)]   # (24, 42)
    W = [[0] * rank for _ in range(p * n)]   # (12, 42)
    
    # Strassen coefficients for 2x2 * 2x2:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    # Result:
    # R00 = P1 + P4 - P5 + P7
    # R01 = P3 + P5
    # R10 = P2 + P4
    # R11 = P1 - P2 + P3 + P6
    
    # U_coeffs[s][(i_loc, j_loc)] = coeff for the s-th Strassen product
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # P1: A00 + A11
        {(1, 0): 1, (1, 1): 1},       # P2: A10 + A11
        {(0, 0): 1},                  # P3: A00
        {(1, 1): 1},                  # P4: A11
        {(0, 0): 1, (0, 1): 1},       # P5: A00 + A01
        {(1, 0): 1, (0, 0): -1},      # P6: A10 - A00
        {(0, 1): 1, (1, 1): -1},      # P7: A01 - A11
    ]
    
    # V_coeffs[s][(j_loc, k_loc)] = coeff for the s-th Strassen product
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # P1: B00 + B11
        {(0, 0): 1},                  # P2: B00
        {(0, 1): 1, (1, 1): -1},      # P3: B01 - B11
        {(1, 0): 1, (0, 0): -1},      # P4: B10 - B00
        {(1, 1): 1},                  # P5: B11
        {(0, 0): 1, (0, 1): 1},       # P6: B00 + B01
        {(1, 0): 1, (1, 1): 1},       # P7: B10 + B11
    ]
    
    # W_contrib[s][(i_loc, k_loc)] = coeff for the s-th Strassen product's contribution to output
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},       # P1: R00(+1), R11(+1)
        {(1, 0): 1, (1, 1): -1},      # P2: R10(+1), R11(-1)
        {(0, 1): 1, (1, 1): 1},       # P3: R01(+1), R11(+1)
        {(0, 0): 1, (1, 0): 1},       # P4: R00(+1), R10(+1)
        {(0, 0): -1, (0, 1): 1},      # P5: R00(-1), R01(+1)
        {(1, 1): 1},                  # P6: R11(+1)
        {(0, 0): 1},                  # P7: R00(+1)
    ]
    
    col = 0
    # For each A-block (a in {0,1}) and B-column-block (b in {0,1,2}) and Strassen term (s in 0..6)
    for a in range(2):       # A block: a=0 for columns 0,1; a=1 for columns 2,3
        for b in range(3):   # B column block: b=0 for cols 0,1; b=1 for cols 2,3; b=2 for cols 4,5
            for s in range(7):  # Strassen term
                # Update U: A block a, Strassen term s
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V: B block (a, b), Strassen term s
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W: output block b, Strassen term s
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
