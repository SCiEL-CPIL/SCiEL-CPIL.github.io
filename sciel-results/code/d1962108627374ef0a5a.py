import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank-minimal decomposition attempt:
    # - 3 Strassen blocks (2x2x2) for k pairs (0,1), (2,3), (4,5) -> Rank 21
    # - 1 Schoolbook block (2x2x1) for k=6, j=0,1 -> Rank 4
    # - 2 terms for j=2 (only k=2 is non-zero) -> Rank 2
    # Total Rank = 27
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, signs=None):
        nonlocal col
        if signs is None:
            signs = [1] * len(u_indices)
        for idx, s in zip(u_indices, signs):
            U[idx][col] += s
        for idx, s in zip(v_indices, signs):
            V[idx][col] += s
        for idx, s in zip(w_indices, signs):
            W[idx][col] += s
        col += 1

    def add_strassen_block(k_start):
        # k_start is 0, 2, 4
        # A indices (i*3+j):
        # A00: 0, A01: 1, A10: 3, A11: 4
        # B indices (j*7+k):
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        # C indices (k*2+i):
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term([0, 4], [b00, b11], [c00, c11], [1, 1, 1, 1])
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term([3, 4], [b00], [c10, c11], [1, 1, 1, 1])
        W[c11][col-1] = -1
        
        # M3: A00(B01-B11) -> C01, C11
        add_term([0], [b01, b11], [c01, c11], [1, 1, 1, 1])
        V[b11][col-1] = -1
        
        # M4: A11(B10-B00) -> C00, C10
        add_term([4], [b10, b00], [c00, c10], [1, 1, 1, 1])
        V[b00][col-1] = -1
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term([0, 1], [b11], [c00, c01], [1, 1, 1, 1])
        W[c00][col-1] = -1
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([3, 0], [b00, b01], [c11], [1, 1, 1, 1])
        U[0][col-1] = -1
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([1, 4], [b10, b11], [c00], [1, 1, 1, 1])
        U[4][col-1] = -1

    # Add 3 Strassen blocks for k=0, 2, 4
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Block for k=6 (2x2x1 Schoolbook) for j=0,1
    # C06 = A00 B06 + A01 B16
    # C16 = A10 B06 + A11 B16
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06], [1])
    add_term([1], [b16], [c06], [1])
    add_term([3], [b06], [c16], [1])
    add_term([4], [b16], [c16], [1])
    
    # Block for j=2: only C_{i,2,2} = A_{i,2} * B_{2,2}
    # So T_{0,2,2} = 1, T_{1,2,2} = 1, all other T_{i,2,k} = 0
    # Use 2 terms:
    # Term 1: U[2][r]=1, V[14][r]=1, W[4][r]=1  -> contributes to C_{0,2,2}
    # Term 2: U[5][r]=1, V[14][r]=1, W[5][r]=1  -> contributes to C_{1,2,2}
    # Wait, V index for j=2, k=2 is 2*7+2 = 16, not 14.
    # B index: j*7+k = 2*7+2 = 16
    # C index: k*2+i = 2*2+0 = 4, 2*2+1 = 5
    
    add_term([2], [16], [4], [1])
    add_term([5], [16], [5], [1])
    
    # Verification step
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        raise ValueError("Dimension mismatch")
    for row in U:
        if len(row) != rank: raise ValueError("U width mismatch")
    for row in V:
        if len(row) != rank: raise ValueError("V width mismatch")
    for row in W:
        if len(row) != rank: raise ValueError("W width mismatch")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
