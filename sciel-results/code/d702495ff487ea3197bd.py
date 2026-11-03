"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 -> 4x8.
    
    Uses Strassen's algorithm on 2x2 blocks of the 4x4x4 sub-problem,
    then extends to handle the 8-wide output by treating it as two
    independent 4x4x4 multiplications.
    
    For each 4x4x4 block:
    - Split into 2x2 blocks of 2x2 matrices
    - Use 7 Strassen forms, each requiring 4 scalar multiplications (schoolbook)
    - Total: 7 * 4 = 28 scalar multiplications per 4x4x4
    
    For 4x4x8: two independent 4x4x4 blocks, total rank = 28 * 2 = 56
    
    The output factor W follows the trace convention W[k*n+i, r]
    where k is the column index of B (0..7), i is the row index of A (0..3),
    and r is the rank index (0..55).
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    rank = 56  # 28 per 4x4x4 block * 2 blocks
    
    # Initialize U, V, W as numpy arrays
    U = np.zeros((n * m, rank), dtype=float)  # (16, 56)
    V = np.zeros((m * p, rank), dtype=float)  # (32, 56)
    W = np.zeros((p * n, rank), dtype=float)  # (32, 56)
    
    # Strassen's 7 forms for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M3 + M6 - M5
    
    # Define Strassen forms as lists of (block_row, block_col, coefficient)
    U_blocks = [
        [(0, 0, 1), (1, 1, 1)],      # M1: A00+A11
        [(1, 0, 1), (1, 1, 1)],      # M2: A10+A11
        [(0, 0, 1)],                  # M3: A00
        [(1, 1, 1)],                  # M4: A11
        [(0, 0, 1), (0, 1, 1)],      # M5: A00+A01
        [(1, 0, 1), (0, 0, -1)],     # M6: A10-A00
        [(0, 1, 1), (1, 1, -1)],     # M7: A01-A11
    ]
    
    V_blocks = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B00+B11
        [(0, 0, 1)],                  # M2: B00
        [(1, 0, 1), (1, 1, -1)],     # M3: B10-B11
        [(0, 1, 1), (1, 0, -1)],     # M4: B01-B10
        [(1, 1, 1)],                  # M5: B11
        [(0, 0, 1), (1, 0, 1)],      # M6: B00+B10
        [(0, 1, 1), (1, 1, 1)],      # M7: B01+B11
    ]
    
    # C contributions: for each C-block (ci, cj), list of (form_s, coefficient)
    C_contrib = {
        (0, 0): [(0, 1), (3, -1), (4, 1), (6, -1)],  # M1 - M4 + M5 - M7
        (0, 1): [(2, 1), (3, 1)],                      # M3 + M4
        (1, 0): [(1, 1), (6, 1)],                      # M2 + M7
        (1, 1): [(0, 1), (2, -1), (5, 1), (4, -1)],   # M1 - M3 + M6 - M5
    }
    
    # For each of the two 4-column blocks of B (block 0: cols 0-3, block 1: cols 4-7)
    for block in range(2):
        for s in range(7):  # 7 Strassen forms
            for a in range(2):  # row index within output 2x2 block
                for t in range(2):  # summation index in 2x2 product
                    for b in range(2):  # col index within output 2x2 block
                        rank_idx = block * 28 + s * 4 + a * 2 + t * 2 + b
                        
                        # U vector: coefficients of A in U_s[a,t]
                        # U_s[a,t] = sum of c * A[2*bi+a, 2*bj+t] for (bi,bj,c) in U_blocks[s]
                        for (bi, bj, c) in U_blocks[s]:
                            ai = 2 * bi + a
                            aj = 2 * bj + t
                            U[ai * m + aj][rank_idx] = c
                        
                        # V vector: coefficients of B in V_s[t,b]
                        # V_s[t,b] = sum of c * B[2*bi+t, block*4 + 2*bj+b] for (bi,bj,c) in V_blocks[s]
                        for (bi, bj, c) in V_blocks[s]:
                            vi = 2 * bi + t
                            vj = block * 4 + 2 * bj + b
                            V[vi * p + vj][rank_idx] = c
                        
                        # W vector: contribution to C
                        # M_s[a,b] contributes to C[2*oci+a, 2*ocj+b] with coefficient
                        # W index: k*n + i where k is col of B (=2*ocj+b), i is row of A (=2*oci+a)
                        for (oci, ocj) in C_contrib:
                            for (form_s, coeff) in C_contrib[(oci, ocj)]:
                                if form_s == s:
                                    wi = 2 * oci + a
                                    wj = 2 * ocj + b
                                    W[wj * n + wi][rank_idx] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
