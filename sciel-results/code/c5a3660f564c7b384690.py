"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _strassen_2x2(x00, x01, x10, x11, y00, y01, y10, y11):
    """Return 7 bilinear terms for 2x2 matrix multiplication."""
    M1 = (x00 + x11) * (y00 + y11)
    M2 = (x10 + x11) * y00
    M3 = x00 * (y10 + y11)
    M4 = x11 * (y01 - y11)
    M5 = (x00 + x01) * y11
    M6 = (x10 - x00) * (y00 + y01)
    M7 = (x01 - x11) * (y10 + y11)
    return [M1, M2, M3, M4, M5, M6, M7]

def construct():
    n, m, p = DIMENSIONS
    # Split A (2x4) into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    # Split B (4x6) into B1 (4x2, cols 0-1), B2 (4x2, cols 2-3), B3 (4x2, cols 4-5)
    # Each Bj splits into Bj_top (2x2, rows 0-1) and Bj_bot (2x2, rows 2-3)
    # C_j = A1 @ Bj_top + A2 @ Bj_bot, each 2x2 product uses Strassen rank 7
    # Total rank = 3 * 2 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    col = 0
    for j in range(3):  # 3 column blocks of B
        for part in range(2):  # top (rows 0-1) or bot (rows 2-3)
            # A block: part=0 -> A1 (cols 0-1), part=1 -> A2 (cols 2-3)
            a_col_base = part * 2  # 0 or 2
            b_row_base = part * 2  # 0 or 2
            
            # Get Strassen coefficients
            # X = A block: X[i, a_col_base + j_local] where i in {0,1}, j_local in {0,1}
            # Y = B block: Y[b_row_base + i_local, j * 2 + k_local] where i_local in {0,1}, k_local in {0,1}
            
            # Define X entries as (U_index, coeff)
            # X00 = A[0, a_col_base + 0] = U index (0 * m + (a_col_base + 0))
            # X01 = A[0, a_col_base + 1] = U index (0 * m + (a_col_base + 1))
            # X10 = A[1, a_col_base + 0] = U index (1 * m + (a_col_base + 0))
            # X11 = A[1, a_col_base + 1] = U index (1 * m + (a_col_base + 1))
            
            x00_u = 0 * m + (a_col_base + 0)
            x01_u = 0 * m + (a_col_base + 1)
            x10_u = 1 * m + (a_col_base + 0)
            x11_u = 1 * m + (a_col_base + 1)
            
            # Y entries:
            # Y00 = B[b_row_base + 0, j*2 + 0] = V index ((b_row_base + 0) * p + (j*2 + 0))
            # Y01 = B[b_row_base + 0, j*2 + 1] = V index ((b_row_base + 0) * p + (j*2 + 1))
            # Y10 = B[b_row_base + 1, j*2 + 0] = V index ((b_row_base + 1) * p + (j*2 + 0))
            # Y11 = B[b_row_base + 1, j*2 + 1] = V index ((b_row_base + 1) * p + (j*2 + 1))
            
            y00_v = (b_row_base + 0) * p + (j * 2 + 0)
            y01_v = (b_row_base + 0) * p + (j * 2 + 1)
            y10_v = (b_row_base + 1) * p + (j * 2 + 0)
            y11_v = (b_row_base + 1) * p + (j * 2 + 1)
            
            # Strassen terms
            # M1 = (X00+X11)*(Y00+Y11)
            # M2 = (X10+X11)*Y00
            # M3 = X00*(Y10+Y11)
            # M4 = X11*(Y01-Y11)
            # M5 = (X00+X01)*Y11
            # M6 = (X10-X00)*(Y00+Y01)
            # M7 = (X01-X11)*(Y10+Y11)
            
            # For each Strassen term, we add one column to U, V, W
            # U entries: coefficients for X entries
            # V entries: coefficients for Y entries
            # W entries: output C[i, j*2 + k] -> W index (k*n + i)
            
            strassen_terms = [
                # (U_coeffs: {u_idx: coeff}, V_coeffs: {v_idx: coeff}, W_coeffs: {w_idx: coeff})
                # M1 = (X00+X11)*(Y00+Y11)
                ({x00_u: 1, x11_u: 1}, {y00_v: 1, y11_v: 1},
                 {(0*2+0)*n+0: 1, (0*2+1)*n+0: -1, (0*2+0)*n+1: 1, (0*2+1)*n+1: 1}),  # C00: +1, C01: -1, C10: +1, C11: +1
                # M2 = (X10+X11)*Y00
                ({x10_u: 1, x11_u: 1}, {y00_v: 1},
                 {(0*2+0)*n+1: 1, (0*2+1)*n+1: -1, (0*2+0)*n+0: 0, (0*2+1)*n+0: 0}),  # C01: +1, C11: -1
                # M3 = X00*(Y10+Y11)
                ({x00_u: 1}, {y10_v: 1, y11_v: 1},
                 {(0*2+1)*n+0: 1, (0*2+1)*n+1: 1, (0*2+0)*n+0: 0, (0*2+0)*n+1: 0}),  # C01: +1, C11: +1
                # M4 = X11*(Y01-Y11)
                ({x11_u: 1}, {y01_v: 1, y11_v: -1},
                 {(0*2+0)*n+0: 1, (0*2+1)*n+1: 1, (0*2+0)*n+1: 0, (0*2+1)*n+0: 0}),  # C00: +1, C11: +1
                # M5 = (X00+X01)*Y11
                ({x00_u: 1, x01_u: 1}, {y11_v: 1},
                 {(0*2+0)*n+0: -1, (0*2+1)*n+0: 1, (0*2+0)*n+1: 0, (0*2+1)*n+1: 0}),  # C00: -1, C10: +1
                # M6 = (X10-X00)*(Y00+Y01)
                ({x10_u: 1, x00_u: -1}, {y00_v: 1, y01_v: 1},
                 {(0*2+0)*n+1: 0, (0*2+1)*n+1: 0, (0*2+0)*n+0: 0, (0*2+1)*n+0: 1}),  # C10: +1
                # M7 = (X01-X11)*(Y10+Y11)
                ({x01_u: 1, x11_u: -1}, {y10_v: 1, y11_v: 1},
                 {(0*2+0)*n+0: 1, (0*2+1)*n+1: 0, (0*2+0)*n+1: 0, (0*2+1)*n+0: 0}),  # C00: +1
            ]
            
            for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                for u_idx, coeff in u_coeffs.items():
                    U[u_idx][col] += coeff
                for v_idx, coeff in v_coeffs.items():
                    V[v_idx][col] += coeff
                for w_idx, coeff in w_coeffs.items():
                    W[w_idx][col] += coeff
                col += 1
    
    # Convert to lists of integers/fractions
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
