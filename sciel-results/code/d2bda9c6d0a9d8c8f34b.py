"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 4
    matrix by a 4 by 6 matrix.
    
    Strategy: Use Strassen-based rank-42 decomposition.
    Split A (2x4) into two 2x2 column blocks.
    Split B (4x6) into two 2x3 row blocks, each with 3 column-pairs of width 2.
    Each (2x2)*(2x2) subproduct uses 7 Strassen terms.
    Total: 2 (A-blocks) * 3 (B-col-blocks) * 7 (Strassen) = 42.
    
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    rank = 42
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (12, 42)
    
    # Strassen coefficients for 2x2 * 2x2 -> 2x2
    # U_coeffs[s]: dict {(i_loc, j_loc): coeff} for first matrix
    # V_coeffs[s]: dict {(j_loc, k_loc): coeff} for second matrix  
    # W_coeffs[s]: dict {(i_loc, k_loc): coeff} for output
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # M1 = (P00+P11)
        {(1, 0): 1, (1, 1): 1},       # M2 = (P10+P11)
        {(0, 0): 1},                   # M3 = P00
        {(1, 1): 1},                   # M4 = P11
        {(0, 0): 1, (0, 1): 1},       # M5 = (P00+P01)
        {(1, 0): 1, (0, 0): -1},      # M6 = (P10-P00)
        {(0, 1): 1, (1, 1): -1},      # M7 = (P01-P11)
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # (Q00+Q11)
        {(0, 0): 1},                   # Q00
        {(0, 1): 1, (1, 1): -1},      # (Q01-Q11)
        {(1, 0): 1, (0, 0): -1},      # (Q10-Q00)
        {(1, 1): 1},                   # Q11
        {(0, 0): 1, (0, 1): 1},       # (Q00+Q01)
        {(1, 0): 1, (1, 1): 1},       # (Q10+Q11)
    ]
    
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # M1 -> C00(+1), C11(+1)
        {(1, 0): 1, (1, 1): -1},      # M2 -> C10(+1), C11(-1)
        {(0, 1): 1, (1, 1): 1},       # M3 -> C01(+1), C11(+1)
        {(0, 0): 1, (1, 0): 1},       # M4 -> C00(+1), C10(+1)
        {(0, 0): -1, (0, 1): 1},      # M5 -> C00(-1), C01(+1)
        {(1, 1): 1},                   # M6 -> C11(+1)
        {(0, 0): 1},                   # M7 -> C00(+1)
    ]
    
    col = 0
    for a_block in range(2):  # A column block: 0 for cols {0,1}, 1 for cols {2,3}
        for b_col_block in range(3):  # B column block: 0 for cols {0,1}, 1 for {2,3}, 2 for {4,5}
            for s in range(7):  # Strassen term
                # Fill U
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Fill V
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Fill W
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
