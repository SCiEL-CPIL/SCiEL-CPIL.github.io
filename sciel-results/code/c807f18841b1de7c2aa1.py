import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Define indices for A (2x4), B (4x4), C (2x4)
    # A: row-major 0..7
    # B: row-major 0..15
    # C: row-major 0..7
    
    # Block definitions: (X_indices, Y_indices, Z_indices)
    # X is 2x2, Y is 2x2, Z is 2x2
    
    # A_L: rows 0,1 of A; cols 0,1
    A_L = {'X00': 0, 'X01': 1, 'X10': 4, 'X11': 5}
    # A_R: rows 0,1 of A; cols 2,3
    A_R = {'X00': 2, 'X01': 3, 'X10': 6, 'X11': 7}
    
    # B_LL: rows 0,1; cols 0,1
    B_LL = {'Y00': 0, 'Y01': 1, 'Y10': 4, 'Y11': 5}
    # B_LR: rows 0,1; cols 2,3
    B_LR = {'Y00': 2, 'Y01': 3, 'Y10': 6, 'Y11': 7}
    # B_RL: rows 2,3; cols 0,1
    B_RL = {'Y00': 8, 'Y01': 9, 'Y10': 12, 'Y11': 13}
    # B_RR: rows 2,3; cols 2,3
    B_RR = {'Y00': 10, 'Y01': 11, 'Y10': 14, 'Y11': 15}
    
    # C_L: rows 0,1 of C; cols 0,1
    C_L = {'Z00': 0, 'Z01': 1, 'Z10': 4, 'Z11': 5}
    # C_R: rows 0,1 of C; cols 2,3
    C_R = {'Z00': 2, 'Z01': 3, 'Z10': 6, 'Z11': 7}
    
    # Strassen terms: (X_coeffs, Y_coeffs)
    # Keys: X00, X01, X10, X11 and Y00, Y01, Y10, Y11
    strassen_terms = [
        # M1
        ({'X00': 1, 'X11': 1}, {'Y00': 1, 'Y11': 1}),
        # M2
        ({'X10': 1, 'X11': 1}, {'Y00': 1}),
        # M3
        ({'X00': 1}, {'Y10': 1, 'Y11': -1}),
        # M4
        ({'X11': 1}, {'Y01': 1, 'Y10': -1}),
        # M5
        ({'X00': 1, 'X01': 1}, {'Y11': 1}),
        # M6
        ({'X10': 1, 'X00': -1}, {'Y00': 1, 'Y10': 1}),
        # M7
        ({'X01': 1, 'X11': -1}, {'Y01': 1, 'Y11': 1}),
    ]
    
    # Output combination for Z from M1..M7
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    out_comb = {
        'Z00': [(0, 1), (3, 1), (4, -1), (6, 1)],
        'Z01': [(2, 1), (4, 1)],
        'Z10': [(1, 1), (3, 1)],
        'Z11': [(0, 1), (1, -1), (2, 1), (5, 1)],
    }
    
    # List of (X_block, Y_block, Z_block)
    products = [
        (A_L, B_LL, C_L),
        (A_R, B_RL, C_L),
        (A_L, B_LR, C_R),
        (A_R, B_RR, C_R),
    ]
    
    col = 0
    for X_idx, Y_idx, Z_idx in products:
        for x_coeffs, y_coeffs in strassen_terms:
            # Fill U
            for name, coeff in x_coeffs.items():
                U[X_idx[name]][col] += coeff
            # Fill V
            for name, coeff in y_coeffs.items():
                V[Y_idx[name]][col] += coeff
            # Fill W
            for z_name, terms in out_comb.items():
                for m_idx, sign in terms:
                    if m_idx == strassen_terms.index((x_coeffs, y_coeffs)):
                        W[Z_idx[z_name]][col] += sign
            col += 1
            
    return {'U': U, 'V': V, 'W': W}
