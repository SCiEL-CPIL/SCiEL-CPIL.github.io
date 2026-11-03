import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Use 3x3 blocks of 2x2 matrices
    # Schoolbook for 3x3 block multiplication: 9 block products
    # Strassen for each 2x2 product: 7 scalar products
    # Total rank: 9 * 7 = 63
    
    rank = 63
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Strassen terms for 2x2: (coeff_A, coeff_B, result_pos)
    # Each term: list of (row, col, coeff) for A, (row, col, coeff) for B, and (row, col, coeff) for C
    
    strassen_2x2 = [
        # M1 = (A00+A11)(B00+B11)
        {
            'A': [(0,0,1), (1,1,1)],
            'B': [(0,0,1), (1,1,1)],
            'C': [(0,0,1), (1,1,1)]
        },
        # M2 = (A10+A11)B00
        {
            'A': [(1,0,1), (1,1,1)],
            'B': [(0,0,1)],
            'C': [(1,0,1)]
        },
        # M3 = A00(B01-B11)
        {
            'A': [(0,0,1)],
            'B': [(0,1,1), (1,1,-1)],
            'C': [(0,1,1)]
        },
        # M4 = A11(B10-B00)
        {
            'A': [(1,1,1)],
            'B': [(1,0,1), (0,0,-1)],
            'C': [(0,0,1), (1,0,1)]
        },
        # M5 = (A00+A01)B11
        {
            'A': [(0,0,1), (0,1,1)],
            'B': [(1,1,1)],
            'C': [(0,0,-1), (0,1,1)]
        },
        # M6 = (A10-A00)(B00+B01)
        {
            'A': [(1,0,1), (0,0,-1)],
            'B': [(0,0,1), (0,1,1)],
            'C': [(1,1,1)]
        },
        # M7 = (A01-A11)(B10+B11)
        {
            'A': [(0,1,1), (1,1,-1)],
            'B': [(1,0,1), (1,1,1)],
            'C': [(0,0,1), (1,1,-1)]
        },
    ]
    
    # For each of the 9 block positions (bi, bk) in A and (bk, bj) in B:
    # A block at (bi, bk) occupies rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
    # B block at (bk, bj) occupies rows 2*bk..2*bk+1, cols 2*bj..2*bj+1
    # C block at (bi, bj) occupies rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    
    idx = 0
    for bi in range(3):  # block row in A, block row in C
        for bk in range(3):  # block col in A, block row in B
            for bj in range(3):  # block col in B, block col in C
                for s in strassen_2x2:
                    # A entries: block (bi, bk), local (r, c)
                    for (r, c, coeff) in s['A']:
                        global_row = 2 * bi + r
                        global_col = 2 * bk + c
                        U[global_row * m + global_col, idx] += coeff
                    
                    # B entries: block (bk, bj), local (r, c)
                    for (r, c, coeff) in s['B']:
                        global_row = 2 * bk + r
                        global_col = 2 * bj + c
                        V[global_row * p + global_col, idx] += coeff
                    
                    # C entries: block (bi, bj), local (r, c)
                    # W indexed as W[col*n + row] per trace convention
                    for (r, c, coeff) in s['C']:
                        global_row = 2 * bi + r
                        global_col = 2 * bj + c
                        W[global_col * n + global_row, idx] += coeff
                    
                    idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
