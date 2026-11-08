import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # Part 1: A[0:2, 0:2] * B[0:2, 0:7] -> C[0:2, 0:7]
    # This is 2x2x7 multiplication.
    # Decomposed into 3 Strassen blocks (2x2x2) for k=0,1; 2,3; 4,5
    # and 1 Schoolbook block (2x2x1) for k=6.
    # Rank = 3 * 7 + 4 = 25.
    
    # Part 2: A[0:2, 2] * B[2, 0:7] -> C[0:2, 0:7]
    # This is 2x1x7 multiplication.
    # Rank = 2 * 1 * 7 = 14.
    
    # Total Rank = 25 + 14 = 39.
    
    rank = 39
    
    # Initialize U, V, W as lists of lists for easy manipulation
    # U shape: (n*m, rank) = (6, 39)
    # V shape: (m*p, rank) = (21, 39)
    # W shape: (p*n, rank) = (14, 39)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank overflow")
        for idx, coeff in zip(u_indices, u_coeffs):
            U[idx][col] += coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            V[idx][col] += coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            W[idx][col] += coeff
        col += 1

    def add_strassen_block(k_start):
        # A indices (2x2 block of 2x3 matrix, rows 0,1; cols 0,1)
        # A00=0, A01=1, A10=3, A11=4
        a00, a01, a10, a11 = 0, 1, 3, 4
        
        # B indices (2x2 block of 3x7 matrix, rows 0,1; cols k_start, k_start+1)
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        # C indices (2x2 block of 2x7 output, rows 0,1; cols k_start, k_start+1)
        # Trace convention: W[k*n+i, r] -> index = k*2 + i
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
        
        # M3: A00(B01-B11) -> C01, C11
        add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])
        
        # M4: A11(B10-B00) -> C00, C10
        add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])

    # Add 3 Strassen blocks for k=0, 2, 4
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Block for k=6: Schoolbook 2x2x1
    # C[i, 6] = sum_j A[i, j] B[j, 6] for j=0,1
    # A00 B06 -> C06
    # A01 B16 -> C06
    # A10 B06 -> C16
    # A11 B16 -> C16
    
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [1], [b06], [1], [c06], [1])
    add_term([1], [1], [b16], [1], [c06], [1])
    add_term([3], [1], [b06], [1], [c16], [1])
    add_term([4], [1], [b16], [1], [c16], [1])
    
    # Part 2: j=2
    # A[i, 2] B[2, k] -> C[i, k]
    # A[0, 2] is index 2, A[1, 2] is index 5.
    # B[2, k] is index 14 + k.
    # C[i, k] is index 2k + i.
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        # A02 B2k -> C0k
        add_term([2], [1], [b2k], [1], [c0k], [1])
        # A12 B2k -> C1k
        add_term([5], [1], [b2k], [1], [c1k], [1])
        
    # Convert to lists for return
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
