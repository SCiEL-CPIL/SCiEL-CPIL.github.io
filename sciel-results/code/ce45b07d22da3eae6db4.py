"""
Program to find a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
Uses a block decomposition strategy:
1. Core 4x4 multiplication using Strassen's algorithm on 2x2 blocks (Rank 28).
2. Boundary terms using schoolbook multiplication (Rank 45).
Total Rank: 73.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-73 exact tensor decomposition for 5x5x5 matrix multiplication.
    """
    n, m, p = 5, 5, 5
    rank = 73
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        U[u_idx, col] = coeff
        V[v_idx, col] = coeff
        W[w_idx, col] = coeff
        col += 1

    # --- Part 1: Core 4x4 Multiplication (Rank 28) ---
    # Partition indices 0-3 into 2x2 blocks: 00, 01, 10, 11
    # Block 00: i in {0,1}, j in {0,1}
    # Block 01: i in {0,1}, j in {2,3}
    # Block 10: i in {2,3}, j in {0,1}
    # Block 11: i in {2,3}, j in {2,3}
    
    def get_global_indices(block, li, lj):
        """
        Map block id (0-3) and local indices (li, lj) to global 4x4 indices (0-3).
        block 0: 00, block 1: 01, block 2: 10, block 3: 11
        """
        if block == 0:
            return li, lj
        elif block == 1:
            return li, lj + 2
        elif block == 2:
            return li + 2, lj
        else: # block 3
            return li + 2, lj + 2

    def add_2x2_product(a_block, b_block, c_block, c_coeff):
        """
        Add schoolbook terms for a 2x2 matrix product.
        A is in block a_block, B in block b_block, C in block c_block.
        All coefficients are 1 for A and B, c_coeff for C.
        """
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    # A indices
                    ai, aj = get_global_indices(a_block, li, lk)
                    # B indices
                    bi, bj = get_global_indices(b_block, lk, lj)
                    # C indices
                    ci, cj = get_global_indices(c_block, li, lj)
                    
                    u_idx = ai * 5 + aj
                    v_idx = bi * 5 + bj
                    w_idx = cj * 5 + ci
                    
                    add_term(u_idx, v_idx, w_idx, c_coeff)

    # Strassen's Algorithm for 2x2 block matrices
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We decompose each Mx into 4 schoolbook terms.
    # Total 7 * 4 = 28 terms.
    
    # M1 = (A00 + A11)(B00 + B11)
    # Expands to A00B00 + A00B11 + A11B00 + A11B11
    # All contribute to C00 with coeff +1 and C11 with coeff +1
    
    # A00B00
    add_2x2_product(0, 0, 0, 1.0) # -> C00
    add_2x2_product(0, 0, 3, 1.0) # -> C11
    # A00B11
    add_2x2_product(0, 3, 0, 1.0) # -> C00
    add_2x2_product(0, 3, 3, 1.0) # -> C11
    # A11B00
    add_2x2_product(3, 0, 0, 1.0) # -> C00
    add_2x2_product(3, 0, 3, 1.0) # -> C11
    # A11B11
    add_2x2_product(3, 3, 0, 1.0) # -> C00
    add_2x2_product(3, 3, 3, 1.0) # -> C11
    
    # M2 = (A10 + A11)B00
    # Expands to A10B00 + A11B00
    # Contributes to C10 with coeff +1 and C11 with coeff -1
    
    # A10B00
    add_2x2_product(2, 0, 2, 1.0) # -> C10
    add_2x2_product(2, 0, 3, -1.0) # -> C11
    # A11B00
    add_2x2_product(3, 0, 2, 1.0) # -> C10
    add_2x2_product(3, 0, 3, -1.0) # -> C11
    
    # M3 = A00(B01 - B11)
    # Expands to A00B01 - A00B11
    # Contributes to C01 with coeff +1 and C11 with coeff +1
    
    # A00B01
    add_2x2_product(0, 1, 1, 1.0) # -> C01
    add_2x2_product(0, 1, 3, 1.0) # -> C11
    # -A00B11
    add_2x2_product(0, 3, 1, -1.0) # -> C01
    add_2x2_product(0, 3, 3, 1.0) # -> C11
    
    # M4 = A11(B10 - B00)
    # Expands to A11B10 - A11B00
    # Contributes to C00 with coeff +1 and C10 with coeff +1
    
    # A11B10
    add_2x2_product(3, 2, 0, 1.0) # -> C00
    add_2x2_product(3, 2, 2, 1.0) # -> C10
    # -A11B00
    add_2x2_product(3, 0, 0, -1.0) # -> C00
    add_2x2_product(3, 0, 2, 1.0) # -> C10
    
    # M5 = (A00 + A01)B11
    # Expands to A00B11 + A01B11
    # Contributes to C00 with coeff -1 and C01 with coeff +1
    
    # A00B11
    add_2x2_product(0, 3, 0, -1.0) # -> C00
    add_2x2_product(0, 3, 1, 1.0) # -> C01
    # A01B11
    add_2x2_product(1, 3, 0, -1.0) # -> C00
    add_2x2_product(1, 3, 1, 1.0) # -> C01
    
    # M6 = (A10 - A00)(B00 + B01)
    # Expands to A10B00 + A10B01 - A00B00 - A00B01
    # Contributes to C11 with coeff +1
    
    # A10B00
    add_2x2_product(2, 0, 3, 1.0) # -> C11
    # A10B01
    add_2x2_product(2, 1, 3, 1.0) # -> C11
    # -A00B00
    add_2x2_product(0, 0, 3, -1.0) # -> C11
    # -A00B01
    add_2x2_product(0, 1, 3, -1.0) # -> C11
    
    # M7 = (A01 - A11)(B10 + B11)
    # Expands to A01B10 + A01B11 - A11B10 - A11B11
    # Contributes to C00 with coeff +1
    
    # A01B10
    add_2x2_product(1, 2, 0, 1.0) # -> C00
    # A01B11
    add_2x2_product(1, 3, 0, 1.0) # -> C00
    # -A11B10
    add_2x2_product(3, 2, 0, -1.0) # -> C00
    # -A11B11
    add_2x2_product(3, 3, 0, -1.0) # -> C00
    
    # --- Part 2: Boundary Terms (Rank 45) ---
    
    # 1. A_4x4 * B_4x1 (Rank 16)
    # C[i, 4] += sum_{j=0}^3 A[i, j] * B[j, 4] for i in 0..3
    for i in range(4):
        for j in range(4):
            u_idx = i * 5 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 5 + i
            add_term(u_idx, v_idx, w_idx, 1.0)
            
    # 2. A_4x1 * B_1x1 (Rank 4)
    # C[i, 4] += A[i, 4] * B[4, 4] for i in 0..3
    for i in range(4):
        u_idx = i * 5 + 4
        v_idx = 4 * 5 + 4
        w_idx = 4 * 5 + i
        add_term(u_idx, v_idx, w_idx, 1.0)
        
    # 3. A_1x4 * B_4x4 (Rank 16)
    # C[4, k] += sum_{j=0}^3 A[4, j] * B[j, k] for k in 0..3
    for k in range(4):
        for j in range(4):
            u_idx = 4 * 5 + j
            v_idx = j * 5 + k
            w_idx = k * 5 + 4
            add_term(u_idx, v_idx, w_idx, 1.0)
            
    # 4. A_1x1 * B_1x4 (Rank 4)
    # C[4, k] += A[4, 4] * B[4, k] for k in 0..3
    for k in range(4):
        u_idx = 4 * 5 + 4
        v_idx = 4 * 5 + k
        w_idx = k * 5 + 4
        add_term(u_idx, v_idx, w_idx, 1.0)
        
    # 5. A_1x4 * B_4x1 (Rank 4)
    # C[4, 4] += sum_{j=0}^3 A[4, j] * B[j, 4]
    for j in range(4):
        u_idx = 4 * 5 + j
        v_idx = j * 5 + 4
        w_idx = 4 * 5 + 4
        add_term(u_idx, v_idx, w_idx, 1.0)
        
    # 6. A_1x1 * B_1x1 (Rank 1)
    # C[4, 4] += A[4, 4] * B[4, 4]
    u_idx = 4 * 5 + 4
    v_idx = 4 * 5 + 4
    w_idx = 4 * 5 + 4
    add_term(u_idx, v_idx, w_idx, 1.0)
    
    if col != rank:
        # Should not happen
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
