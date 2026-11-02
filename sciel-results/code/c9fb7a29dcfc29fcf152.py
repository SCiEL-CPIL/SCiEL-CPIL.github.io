"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strategy: Strassen 2x2 block decomposition with 3x3 schoolbook sub-multiplications
    # Partition 6x6 into 2x2 blocks of 3x3 submatrices
    # i = 2*i_b + i_s, where i_b in {0,1} (block row), i_s in {0,1,2} (within block)
    # Total rank = 7 (Strassen) * 27 (3x3 schoolbook) = 189
    
    rank = 7 * 27  # 189
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2 coefficients
    # Block indices: (row_block, col_block) in {0,1}x{0,1}
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    strassen = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3 = A00(B10-B11)
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,0): 1}),
        # M4 = A11(B01-B10)
        ({(1,1): 1}, {(0,1): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)(B00+B10)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        # M7 = (A01-A11)(B01+B11)
        ({(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]
    
    col = 0
    for s in range(7):
        U_coeff = strassen[s][0]
        V_coeff = strassen[s][1]
        W_coeff = strassen[s][2]
        
        # For each of the 27 schoolbook terms within the 3x3 blocks
        for i_s in range(3):  # row within 3x3 block
            for j_s in range(3):  # col within 3x3 block of A
                for k_s in range(3):  # col within 3x3 block of B
                
                    # U: A[i,j] where i = 2*i_b + i_s, j = 2*j_b + j_s
                    for i_b in range(2):
                        for j_b in range(2):
                            c = U_coeff.get((i_b, j_b), 0)
                            if c != 0:
                                i = 2 * i_b + i_s
                                j = 2 * j_b + j_s
                                U[i * m + j, col] += c
                    
                    # V: B[j,k] where j = 2*j_b + j_s, k = 2*k_b + k_s
                    for j_b in range(2):
                        for k_b in range(2):
                            c = V_coeff.get((j_b, k_b), 0)
                            if c != 0:
                                j = 2 * j_b + j_s
                                k = 2 * k_b + k_s
                                V[j * p + k, col] += c
                    
                    # W: C[k,i] where k = 2*k_b + k_s, i = 2*i_b + i_s
                    for k_b in range(2):
                        for i_b in range(2):
                            c = W_coeff.get((k_b, i_b), 0)
                            if c != 0:
                                k = 2 * k_b + k_s
                                i = 2 * i_b + i_s
                                W[k * n + i, col] += c
                    
                    col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
