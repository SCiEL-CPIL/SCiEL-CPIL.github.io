import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The current best known rank for 2x3 by 3x7 is 39.
    # Strategy: 3 Strassen blocks (rank 21) + 1 Schoolbook block for k=6 (rank 4) + 1 Schoolbook block for j=2 (rank 14)
    
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank overflow")
        for idx, coeff in zip(u_indices, u_coeffs):
            U[idx, col] += coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            V[idx, col] += coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            W[idx, col] += coeff
        col += 1

    def add_strassen_block(k_start):
        # A indices (2x2 block of 2x3 matrix, rows 0,1; cols 0,1)
        # A00=0, A01=1, A10=3, A11=4
        a00, a01, a10, a11 = 0, 1, 3, 4
        
        # B indices (2x2 block of 3x7 matrix, rows 0,1; cols k_start, k_start+1)
        # B00 = 0*7 + k_start, B01 = 0*7 + k_start+1
        # B10 = 1*7 + k_start, B11 = 1*7 + k_start+1
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        # C indices (2x2 block of 2x7 output, rows 0,1; cols k_start, k_start+1)
        # Trace convention: W[k*n+i, r] where output is C[i, k]
        # Index = k*2 + i
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
    # C[i, 6] = sum_j A[i, j] B[j, 6]
    # Only j=0,1,2 contribute.
    # The Strassen blocks handled j=0,1 for k=0,1,2,3,4,5.
    # So for k=6, we need j=0,1,2.
    # C[0, 6] = A[0, 0]B[0, 6] + A[0, 1]B[1, 6] + A[0, 2]B[2, 6]
    # C[1, 6] = A[1, 0]B[0, 6] + A[1, 1]B[1, 6] + A[1, 2]B[2, 6]
    
    # The j=0,1 part for k=6 is 2x2x1 schoolbook (rank 4)
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    # A00 B06 -> C06
    add_term([0], [1], [b06], [1], [c06], [1])
    # A01 B16 -> C06
    add_term([1], [1], [b16], [1], [c06], [1])
    # A10 B06 -> C16
    add_term([3], [1], [b06], [1], [c16], [1])
    # A11 B16 -> C16
    add_term([4], [1], [b16], [1], [c16], [1])
    
    # The j=2 part for all k=0..6
    # C[i, k] += A[i, 2] B[2, k]
    # This is a 2x1 by 1x7 multiplication.
    # A[0, 2] is index 2, A[1, 2] is index 5.
    # B[2, k] is index 2*7 + k = 14 + k.
    # For each k, we have 2 terms: A02 B2k -> C0k, A12 B2k -> C1k.
    # Total 14 terms.
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        # A02 B2k -> C0k
        add_term([2], [1], [b2k], [1], [c0k], [1])
        # A12 B2k -> C1k
        add_term([5], [1], [b2k], [1], [c1k], [1])
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
