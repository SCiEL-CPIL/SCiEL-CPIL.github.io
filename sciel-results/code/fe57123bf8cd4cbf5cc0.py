import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=4
    
    # We implement the Strassen-based decomposition for 4x4 matrices.
    # This decomposes 4x4 into a 2x2 block structure, where each block is 2x2.
    # Then each 2x2 block multiplication is decomposed using Strassen's 7-term algorithm.
    # Total rank = 7 * 7 = 49.
    
    # U shape: (n*m, rank) = (16, 49)
    # V shape: (m*p, rank) = (16, 49)
    # W shape: (p*n, rank) = (16, 49)
    
    # Indexing convention:
    # Matrix A (4x4) is stored as vector of length 16.
    # A[i, j] maps to index i*4 + j.
    # Block (b_row, b_col) of 2x2 blocks:
    #   b_row in {0, 1}, b_col in {0, 1}
    #   Element (i, j) within block maps to global (2*b_row + i, 2*b_col + j)
    #   Global index = (2*b_row + i)*4 + (2*b_col + j)
    
    # Strassen 2x2 decomposition coefficients:
    # For 2x2 matrices X, Y, Z = X*Y
    # P1 = (X00 + X11) * (Y00 + Y11)
    # P2 = (X10 + X11) * Y00
    # P3 = X00 * (Y01 - Y11)
    # P4 = X11 * (Y10 - Y00)
    # P5 = (X00 + X01) * Y11
    # P6 = (X10 - X00) * (Y00 + Y01)
    # P7 = (X01 - X11) * (Y10 + Y11)
    
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # Coefficients for X (U2) and Y (V2) for each term k=1..7:
    # X indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Y indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # U2[k][x_idx], V2[k][y_idx]
    U2 = np.zeros((7, 4), dtype=int)
    V2 = np.zeros((7, 4), dtype=int)
    
    # k=0 (P1)
    U2[0, 0] = 1; U2[0, 3] = 1
    V2[0, 0] = 1; V2[0, 3] = 1
    
    # k=1 (P2)
    U2[1, 2] = 1; U2[1, 3] = 1
    V2[1, 0] = 1
    
    # k=2 (P3)
    U2[2, 0] = 1
    V2[2, 1] = 1; V2[2, 3] = -1
    
    # k=3 (P4)
    U2[3, 3] = 1
    V2[3, 2] = 1; V2[3, 0] = -1
    
    # k=4 (P5)
    U2[4, 0] = 1; U2[4, 1] = 1
    V2[4, 3] = 1
    
    # k=5 (P6)
    U2[5, 2] = 1; U2[5, 0] = -1
    V2[5, 0] = 1; V2[5, 1] = 1
    
    # k=6 (P7)
    U2[6, 1] = 1; U2[6, 3] = -1
    V2[6, 2] = 1; V2[6, 3] = 1
    
    # W2[k][z_idx] coefficients for Z = sum_k W2[k][z_idx] * P_k
    # Z indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W2 = np.zeros((7, 4), dtype=int)
    
    # Z00 = P1 + P4 - P5 + P7
    W2[0, 0] = 1  # P1 contributes to Z00
    W2[3, 0] = 1  # P4 contributes to Z00
    W2[4, 0] = -1 # P5 contributes to Z00
    W2[6, 0] = 1  # P7 contributes to Z00
    
    # Z01 = P3 + P5
    W2[2, 1] = 1  # P3 contributes to Z01
    W2[4, 1] = 1  # P5 contributes to Z01
    
    # Z10 = P2 + P4
    W2[1, 2] = 1  # P2 contributes to Z10
    W2[3, 2] = 1  # P4 contributes to Z10
    
    # Z11 = P1 - P2 + P3 + P6
    W2[0, 3] = 1  # P1 contributes to Z11
    W2[1, 3] = -1 # P2 contributes to Z11
    W2[2, 3] = 1  # P3 contributes to Z11
    W2[5, 3] = 1  # P6 contributes to Z11
    
    # Now construct 4x4 decomposition.
    # Outer level: 2x2 blocks of 2x2 matrices.
    # Inner level: 2x2 scalar matrices.
    
    # U shape: (16, 49)
    # V shape: (16, 49)
    # W shape: (16, 49)
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    # Helper function to get global index for block (b_row, b_col) and local (i, j)
    def global_idx(b_row, b_col, i, j):
        return (2 * b_row + i) * 4 + (2 * b_col + j)
    
    # For each outer term k_outer (0..6) and inner term k_inner (0..6):
    # The term index is k_outer * 7 + k_inner
    
    for k_outer in range(7):
        for k_inner in range(7):
            term_idx = k_outer * 7 + k_inner
            
            # U_outer[k_outer] gives coefficients for A blocks
            # U_inner[k_inner] gives coefficients for sub-blocks within A_k
            # Combined: for each block (br, bc) and sub-block (sr, sc):
            # coeff = U_outer[k_outer, br*2+bc] * U_inner[k_inner, sr*2+sc]
            # Wait, U2 is defined for 2x2 matrices with indices 0,1,2,3 corresponding to (0,0),(0,1),(1,0),(1,1)
            # So block index (br, bc) maps to U2 index br*2 + bc
            # Sub-block index (sr, sc) maps to U2 index sr*2 + sc
            
            # For U (A matrix):
            for br in range(2):  # block row
                for bc in range(2):  # block col
                    outer_coeff = U2[k_outer, br * 2 + bc]
                    if outer_coeff == 0:
                        continue
                    for sr in range(2):  # sub-block row
                        for sc in range(2):  # sub-block col
                            inner_coeff = U2[k_inner, sr * 2 + sc]
                            if inner_coeff == 0:
                                continue
                            coeff = outer_coeff * inner_coeff
                            idx = global_idx(br, bc, sr, sc)
                            U[idx, term_idx] += coeff
            
            # For V (B matrix):
            for br in range(2):  # block row
                for bc in range(2):  # block col
                    outer_coeff = V2[k_outer, br * 2 + bc]
                    if outer_coeff == 0:
                        continue
                    for sr in range(2):  # sub-block row
                        for sc in range(2):  # sub-block col
                            inner_coeff = V2[k_inner, sr * 2 + sc]
                            if inner_coeff == 0:
                                continue
                            coeff = outer_coeff * inner_coeff
                            idx = global_idx(br, bc, sr, sc)
                            V[idx, term_idx] += coeff
            
            # For W (C matrix):
            # W_outer[k_outer] gives coefficients for C blocks
            # W_inner[k_inner] gives coefficients for sub-blocks within C_k
            # Combined: for each block (br, bc) and sub-block (sr, sc):
            # coeff = W_outer[k_outer, br*2+bc] * W_inner[k_inner, sr*2+sc]
            for br in range(2):  # block row
                for bc in range(2):  # block col
                    outer_coeff = W2[k_outer, br * 2 + bc]
                    if outer_coeff == 0:
                        continue
                    for sr in range(2):  # sub-block row
                        for sc in range(2):  # sub-block col
                            inner_coeff = W2[k_inner, sr * 2 + sc]
                            if inner_coeff == 0:
                                continue
                            coeff = outer_coeff * inner_coeff
                            idx = global_idx(br, bc, sr, sc)
                            W[idx, term_idx] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
