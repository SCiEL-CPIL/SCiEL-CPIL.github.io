import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank: 39 (current best known / seed)
    # Structure: 3 Strassen blocks (rank 21) + 1 Schoolbook block for k=6 (rank 4) + 1 Schoolbook block for j=2 (rank 14)
    rank = 39
    
    # Use Fraction for exact rational arithmetic if we were solving, 
    # but here we construct the known valid decomposition directly.
    # The "refinement" idea from inspirations is to verify correctness and ensure minimal rank structure.
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_term(u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank overflow")
        for i, c in zip(u_idx, u_coeff):
            U[i, col] += c
        for i, c in zip(v_idx, v_coeff):
            V[i, col] += c
        for i, c in zip(w_idx, w_coeff):
            W[i, col] += c
        col += 1

    def add_strassen_block(k_start):
        # A indices: 2x2 block of 2x3 matrix (rows 0,1; cols 0,1)
        a00, a01, a10, a11 = 0, 1, 3, 4
        
        # B indices: 2x2 block of 3x7 matrix (rows 0,1; cols k_start, k_start+1)
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        # C indices: 2x2 block of 2x7 output (rows 0,1; cols k_start, k_start+1)
        # Convention: W[k*n+i, r] => index = k*2 + i
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
    # The Strassen blocks handled j=0,1 for k=0,1,2,3,4,5.
    # For k=6, we need j=0,1,2.
    # The j=0,1 part is handled here (rank 4).
    # The j=2 part is handled in the next section (rank 14).
    
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
    
    # Part 2: j=2
    # A02 B2k -> C0k
    # A12 B2k -> C1k
    # This handles the contribution of the 3rd row of B and 3rd col of A for all k.
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        # A02 B2k -> C0k
        add_term([2], [1], [b2k], [1], [c0k], [1])
        # A12 B2k -> C1k
        add_term([5], [1], [b2k], [1], [c1k], [1])
        
    # Ensure we didn't overflow or underflow
    if col != rank:
        # If col < rank, we have extra zero columns. We can truncate.
        # If col > rank, it's an error.
        if col > rank:
            raise ValueError("Rank exceeded")
        else:
            U = U[:, :col]
            V = V[:, :col]
            W = W[:, :col]
            rank = col

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
