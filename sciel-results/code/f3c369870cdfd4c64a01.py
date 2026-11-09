"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    # Rank 22 decomposition
    # This is a known correct decomposition for 2x3x4 matrix multiplication.
    # It uses Strassen's algorithm for the two 2x2 blocks of the multiplication
    # and schoolbook multiplication for the remaining 2x1x4 block.
    
    rank = 22
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (6, 22)
    # V: (m*p, rank) = (12, 22)
    # W: (p*n, rank) = (8, 22)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a term
    # u_indices: list of row indices in U
    # v_indices: list of row indices in V
    # w_indices: list of row indices in W
    # col: the column in U, V, W
    # coeffs_u, coeffs_v, coeffs_w: coefficients for each index
    
    def add_term(u_indices, v_indices, w_indices, col, coeffs_u, coeffs_v, coeffs_w):
        for idx, coeff in zip(u_indices, coeffs_u):
            U[idx, col] += coeff
        for idx, coeff in zip(v_indices, coeffs_v):
            V[idx, col] += coeff
        for idx, coeff in zip(w_indices, coeffs_w):
            W[idx, col] += coeff
            
    # Indices for A (2x3): A[i,j] -> i*3+j
    # A00=0, A01=1, A02=2
    # A10=3, A11=4, A12=5
    
    # Indices for B (3x4): B[j,k] -> j*4+k
    # B00=0, B01=1, B02=2, B03=3
    # B10=4, B11=5, B12=6, B13=7
    # B20=8, B21=9, B22=10, B23=11
    
    # Indices for W (p*n = 8): W[k*n+i, col] -> row index is k*n+i
    # C[i,k] corresponds to W[k*n+i]
    # C00 -> W[0]
    # C01 -> W[2]
    # C02 -> W[4]
    # C03 -> W[6]
    # C10 -> W[1]
    # C11 -> W[3]
    # C12 -> W[5]
    # C13 -> W[7]
    
    # Block 1: A_L (2x2) * B_T1 (2x2) -> C1 (2x2)
    # A_L: A00(0), A01(1), A10(3), A11(4)
    # B_T1: B00(0), B01(1), B10(4), B11(5)
    # C1: C00(W0), C01(W2), C10(W1), C11(W3)
    
    # Strassen Terms for Block 1
    # M1: (A00+A11)(B00+B11)
    add_term([0, 4], [0, 5], [0, 3], 0, [1, 1], [1, 1], [1, 1])
    # M2: (A10+A11)B00
    add_term([3, 4], [0], [1, 3], 1, [1, 1], [1], [1, -1])
    # M3: A00(B01-B11)
    add_term([0], [1, 5], [2, 3], 2, [1], [1, -1], [1, 1])
    # M4: A11(B10-B00)
    add_term([4], [4, 0], [0, 1], 3, [1], [1, -1], [1, 1])
    # M5: (A00+A01)B11
    add_term([0, 1], [5], [0, 2], 4, [1, 1], [1], [-1, 1])
    # M6: (A10-A00)(B00+B01)
    add_term([3, 0], [0, 1], [3], 5, [1, -1], [1, 1], [1])
    # M7: (A01-A11)(B10+B11)
    add_term([1, 4], [4, 5], [0], 6, [1, -1], [1, 1], [1])
    
    # Block 2: A_L (2x2) * B_T2 (2x2) -> C2 (2x2)
    # A_L: A00(0), A01(1), A10(3), A11(4)
    # B_T2: B02(2), B03(3), B12(6), B13(7)
    # C2: C02(W4), C03(W6), C12(W5), C13(W7)
    
    # Strassen Terms for Block 2
    # M1: (A00+A11)(B02+B13)
    add_term([0, 4], [2, 7], [4, 7], 7, [1, 1], [1, 1], [1, 1])
    # M2: (A10+A11)B02
    add_term([3, 4], [2], [5, 7], 8, [1, 1], [1], [1, -1])
    # M3: A00(B03-B13)
    add_term([0], [3, 7], [6, 7], 9, [1], [1, -1], [1, 1])
    # M4: A11(B12-B02)
    add_term([4], [6, 2], [4, 5], 10, [1], [1, -1], [1, 1])
    # M5: (A00+A01)B13
    add_term([0, 1], [7], [4, 6], 11, [1, 1], [1], [-1, 1])
    # M6: (A10-A00)(B02+B03)
    add_term([3, 0], [2, 3], [7], 12, [1, -1], [1, 1], [1])
    # M7: (A01-A11)(B12+B13)
    add_term([1, 4], [6, 7], [4], 13, [1, -1], [1, 1], [1])
    
    # Block 3: A_R (2x1) * B_B (1x4) -> C3 (2x4)
    # A_R: A02(2), A12(5)
    # B_B: B20(8), B21(9), B22(10), B23(11)
    # C3: C00(W0), C10(W1), C01(W2), C11(W3), C02(W4), C12(W5), C03(W6), C13(W7)
    
    # Schoolbook Terms for Block 3
    # Z1: A02 B20 -> C00 (W0)
    add_term([2], [8], [0], 14, [1], [1], [1])
    # Z2: A02 B21 -> C01 (W2)
    add_term([2], [9], [2], 15, [1], [1], [1])
    # Z3: A02 B22 -> C02 (W4)
    add_term([2], [10], [4], 16, [1], [1], [1])
    # Z4: A02 B23 -> C03 (W6)
    add_term([2], [11], [6], 17, [1], [1], [1])
    # Z5: A12 B20 -> C10 (W1)
    add_term([5], [8], [1], 18, [1], [1], [1])
    # Z6: A12 B21 -> C11 (W3)
    add_term([5], [9], [3], 19, [1], [1], [1])
    # Z7: A12 B22 -> C12 (W5)
    add_term([5], [10], [5], 20, [1], [1], [1])
    # Z8: A12 B23 -> C13 (W7)
    add_term([5], [11], [7], 21, [1], [1], [1])
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
