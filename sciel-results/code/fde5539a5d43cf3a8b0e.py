import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition by summing contributions from different blocks.
    # Total rank will be the sum of ranks of the components.
    
    # Component 1: C_L = A * B_L (3x3 * 3x3)
    # Split A into 2x2, 2x1, 1x2, 1x1 blocks.
    # Split B_L into 2x2, 2x1, 1x2, 1x1 blocks.
    
    # Indices for A (3x3):
    # A[0,0], A[0,1], A[0,2]
    # A[1,0], A[1,1], A[1,2]
    # A[2,0], A[2,1], A[2,2]
    
    # Block A11 (2x2): A[0,0], A[0,1], A[1,0], A[1,1]
    # Block A12 (2x1): A[0,2], A[1,2]
    # Block A21 (1x2): A[2,0], A[2,1]
    # Block A22 (1x1): A[2,2]
    
    # Block B_L11 (2x2): B[0,0], B[0,1], B[1,0], B[1,1]
    # Block B_L12 (2x1): B[0,2], B[1,2]
    # Block B_L21 (1x2): B[2,0], B[2,1]
    # Block B_L22 (1x1): B[2,2]
    
    # C_L11 (2x2) = A11 * B_L11 (Rank 7 Strassen)
    # C_L12 (2x1) = A11 * B_L12 + A12 * B_L22
    # C_L21 (1x2) = A21 * B_L11 + A22 * B_L21
    # C_L22 (1x1) = A21 * B_L12 + A22 * B_L22
    
    # Component 2: C_R = A * B_R (3x3 * 3x2)
    # B_R is 3x2.
    # Split B_R into top (2x2) and bot (1x2).
    # B_R_top: B[0,3], B[0,4], B[1,3], B[1,4]
    # B_R_bot: B[2,3], B[2,4]
    
    # C_R_top (2x2) = A11 * B_R_top + A12 * B_R_bot
    # C_R_bot (1x2) = A21 * B_R_top + A22 * B_R_bot
    
    # We need to create U, V, W matrices.
    # U shape: (9, R)
    # V shape: (15, R)
    # W shape: (15, R)
    
    # We will accumulate the rank contributions.
    # Let's define a helper to add a rank-1 term.
    
    R = 0
    U = []
    V = []
    W = []
    
    def add_term(u_indices, v_indices, w_indices, coeffs):
        nonlocal R
        # u_indices: list of (idx, coeff) for U
        # v_indices: list of (idx, coeff) for V
        # w_indices: list of (idx, coeff) for W
        # This adds one rank-1 tensor.
        
        u_row = [0.0] * 9
        v_row = [0.0] * 15
        w_row = [0.0] * 15
        
        for idx, c in u_indices:
            u_row[idx] = c
        for idx, c in v_indices:
            v_row[idx] = c
        for idx, c in w_indices:
            w_row[idx] = c
            
        U.append(u_row)
        V.append(v_row)
        W.append(w_row)
        R += 1

    # Helper to get global index for A (i, j)
    def a_idx(i, j): return i * 3 + j
    # Helper to get global index for B (j, k)
    def b_idx(j, k): return j * 5 + k
    # Helper to get global index for C (i, k) -> W index is k*3 + i
    def c_idx(i, k): return k * 3 + i

    # --- Strassen 2x2 (Rank 7) ---
    # Inputs: A11 (2x2), B11 (2x2)
    # A11 indices: (0,0), (0,1), (1,0), (1,1)
    # B11 indices: (0,0), (0,1), (1,0), (1,1)
    # Output: C11 (2x2) indices (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map these to the specific blocks.
    
    # 1. C_L11 = A11 * B_L11
    # A11 elements: A[0,0], A[0,1], A[1,0], A[1,1]
    # B_L11 elements: B[0,0], B[0,1], B[1,0], B[1,1]
    # C_L11 elements: C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Term 1: M1
    u = [(a_idx(0,0), 1), (a_idx(1,1), 1)]
    v = [(b_idx(0,0), 1), (b_idx(1,1), 1)]
    w = [(c_idx(0,0), 1), (c_idx(0,1), 1), (c_idx(1,0), 1), (c_idx(1,1), 1)]
    # Wait, Strassen M1 contributes to all 4 outputs.
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1.
    # So W has 1s at all 4 positions.
    add_term(u, v, w, None)
    
    # Term 2: M2
    u = [(a_idx(1,0), 1), (a_idx(1,1), 1)]
    v = [(b_idx(0,0), 1)]
    w = [(c_idx(1,0), 1), (c_idx(1,1), -1)] # C10 += M2, C11 -= M2
    add_term(u, v, w, None)
    
    # Term 3: M3
    u = [(a_idx(0,0), 1)]
    v = [(b_idx(0,1), 1), (b_idx(1,1), -1)]
    w = [(c_idx(0,1), 1), (c_idx(1,1), 1)] # C01 += M3, C11 += M3
    add_term(u, v, w, None)
    
    # Term 4: M4
    u = [(a_idx(1,1), 1)]
    v = [(b_idx(1,0), 1), (b_idx(0,0), -1)]
    w = [(c_idx(0,0), 1), (c_idx(1,0), 1)] # C00 += M4, C10 += M4
    add_term(u, v, w, None)
    
    # Term 5: M5
    u = [(a_idx(0,0), 1), (a_idx(0,1), 1)]
    v = [(b_idx(1,1), 1)]
    w = [(c_idx(0,0), -1), (c_idx(0,1), 1)] # C00 -= M5, C01 += M5
    add_term(u, v, w, None)
    
    # Term 6: M6
    u = [(a_idx(1,0), 1), (a_idx(0,0), -1)]
    v = [(b_idx(0,0), 1), (b_idx(0,1), 1)]
    w = [(c_idx(1,1), 1)] # C11 += M6
    add_term(u, v, w, None)
    
    # Term 7: M7
    u = [(a_idx(0,1), 1), (a_idx(1,1), -1)]
    v = [(b_idx(1,0), 1), (b_idx(1,1), 1)]
    w = [(c_idx(0,0), 1), (c_idx(1,0), -1)] # C00 += M7, C10 -= M7
    add_term(u, v, w, None)

    # 2. C_L12 (2x1) = A11 * B_L12 + A12 * B_L22
    # A11 * B_L12: 2x2 * 2x1. Schoolbook Rank 4.
    # B_L12: B[0,2], B[1,2].
    # A11: A[0,0], A[0,1], A[1,0], A[1,1].
    # C_L12: C[0,2], C[1,2].
    
    # Term 8: A00 * B02
    u = [(a_idx(0,0), 1)]
    v = [(b_idx(0,2), 1)]
    w = [(c_idx(0,2), 1)]
    add_term(u, v, w, None)
    
    # Term 9: A01 * B12
    u = [(a_idx(0,1), 1)]
    v = [(b_idx(1,2), 1)]
    w = [(c_idx(0,2), 1)]
    add_term(u, v, w, None)
    
    # Term 10: A10 * B02
    u = [(a_idx(1,0), 1)]
    v = [(b_idx(0,2), 1)]
    w = [(c_idx(1,2), 1)]
    add_term(u, v, w, None)
    
    # Term 11: A11 * B12
    u = [(a_idx(1,1), 1)]
    v = [(b_idx(1,2), 1)]
    w = [(c_idx(1,2), 1)]
    add_term(u, v, w, None)
    
    # A12 * B_L22: 2x1 * 1x1. Rank 2.
    # A12: A[0,2], A[1,2].
    # B_L22: B[2,2].
    
    # Term 12: A02 * B22
    u = [(a_idx(0,2), 1)]
    v = [(b_idx(2,2), 1)]
    w = [(c_idx(0,2), 1)]
    add_term(u, v, w, None)
    
    # Term 13: A12 * B22
    u = [(a_idx(1,2), 1)]
    v = [(b_idx(2,2), 1)]
    w = [(c_idx(1,2), 1)]
    add_term(u, v, w, None)

    # 3. C_L21 (1x2) = A21 * B_L11 + A22 * B_L21
    # A21 * B_L11: 1x2 * 2x2. Rank 4.
    # A21: A[2,0], A[2,1].
    # B_L11: B[0,0], B[0,1], B[1,0], B[1,1].
    # C_L21: C[2,0], C[2,1].
    
    # Term 14: A20 * B00
    u = [(a_idx(2,0), 1)]
    v = [(b_idx(0,0), 1)]
    w = [(c_idx(2,0), 1)]
    add_term(u, v, w, None)
    
    # Term 15: A20 * B10
    u = [(a_idx(2,0), 1)]
    v = [(b_idx(1,0), 1)]
    w = [(c_idx(2,0), 1)]
    add_term(u, v, w, None)
    
    # Term 16: A21 * B01
    u = [(a_idx(2,1), 1)]
    v = [(b_idx(0,1), 1)]
    w = [(c_idx(2,1), 1)]
    add_term(u, v, w, None)
    
    # Term 17: A21 * B11
    u = [(a_idx(2,1), 1)]
    v = [(b_idx(1,1), 1)]
    w = [(c_idx(2,1), 1)]
    add_term(u, v, w, None)
    
    # A22 * B_L21: 1x1 * 1x2. Rank 2.
    # A22: A[2,2].
    # B_L21: B[2,0], B[2,1].
    
    # Term 18: A22 * B20
    u = [(a_idx(2,2), 1)]
    v = [(b_idx(2,0), 1)]
    w = [(c_idx(2,0), 1)]
    add_term(u, v, w, None)
    
    # Term 19: A22 * B21
    u = [(a_idx(2,2), 1)]
    v = [(b_idx(2,1), 1)]
    w = [(c_idx(2,1), 1)]
    add_term(u, v, w, None)

    # 4. C_L22 (1x1) = A21 * B_L12 + A22 * B_L22
    # A21 * B_L12: 1x2 * 2x1. Rank 2.
    # A21: A[2,0], A[2,1].
    # B_L12: B[0,2], B[1,2].
    
    # Term 20: A20 * B02
    u = [(a_idx(2,0), 1)]
    v = [(b_idx(0,2), 1)]
    w = [(c_idx(2,2), 1)]
    add_term(u, v, w, None)
    
    # Term 21: A21 * B12
    u = [(a_idx(2,1), 1)]
    v = [(b_idx(1,2), 1)]
    w = [(c_idx(2,2), 1)]
    add_term(u, v, w, None)
    
    # A22 * B_L22: 1x1 * 1x1. Rank 1.
    # Term 22: A22 * B22
    u = [(a_idx(2,2), 1)]
    v = [(b_idx(2,2), 1)]
    w = [(c_idx(2,2), 1)]
    add_term(u, v, w, None)

    # --- Component 2: C_R = A * B_R (3x3 * 3x2) ---
    
    # 5. C_R_top (2x2) = A11 * B_R_top + A12 * B_R_bot
    # A11 * B_R_top: 2x2 * 2x2. Rank 7 (Strassen).
    # B_R_top: B[0,3], B[0,4], B[1,3], B[1,4].
    
    # Term 23: M1
    u = [(a_idx(0,0), 1), (a_idx(1,1), 1)]
    v = [(b_idx(0,3), 1), (b_idx(1,4), 1)]
    w = [(c_idx(0,3), 1), (c_idx(0,4), 1), (c_idx(1,3), 1), (c_idx(1,4), 1)]
    add_term(u, v, w, None)
    
    # Term 24: M2
    u = [(a_idx(1,0), 1), (a_idx(1,1), 1)]
    v = [(b_idx(0,3), 1)]
    w = [(c_idx(1,3), 1), (c_idx(1,4), -1)]
    add_term(u, v, w, None)
    
    # Term 25: M3
    u = [(a_idx(0,0), 1)]
    v = [(b_idx(0,4), 1), (b_idx(1,4), -1)]
    w = [(c_idx(0,4), 1), (c_idx(1,4), 1)]
    add_term(u, v, w, None)
    
    # Term 26: M4
    u = [(a_idx(1,1), 1)]
    v = [(b_idx(1,3), 1), (b_idx(0,3), -1)]
    w = [(c_idx(0,3), 1), (c_idx(1,3), 1)]
    add_term(u, v, w, None)
    
    # Term 27: M5
    u = [(a_idx(0,0), 1), (a_idx(0,1), 1)]
    v = [(b_idx(1,4), 1)]
    w = [(c_idx(0,3), -1), (c_idx(0,4), 1)]
    add_term(u, v, w, None)
    
    # Term 28: M6
    u = [(a_idx(1,0), 1), (a_idx(0,0), -1)]
    v = [(b_idx(0,3), 1), (b_idx(0,4), 1)]
    w = [(c_idx(1,4), 1)]
    add_term(u, v, w, None)
    
    # Term 29: M7
    u = [(a_idx(0,1), 1), (a_idx(1,1), -1)]
    v = [(b_idx(1,3), 1), (b_idx(1,4), 1)]
    w = [(c_idx(0,3), 1), (c_idx(1,3), -1)]
    add_term(u, v, w, None)
    
    # A12 * B_R_bot: 2x1 * 1x2. Rank 2.
    # A12: A[0,2], A[1,2].
    # B_R_bot: B[2,3], B[2,4].
    
    # Term 30: A02 * B23
    u = [(a_idx(0,2), 1)]
    v = [(b_idx(2,3), 1)]
    w = [(c_idx(0,3), 1)]
    add_term(u, v, w, None)
    
    # Term 31: A02 * B24
    u = [(a_idx(0,2), 1)]
    v = [(b_idx(2,4), 1)]
    w = [(c_idx(0,4), 1)]
    add_term(u, v, w, None)
    
    # Term 32: A12 * B23
    u = [(a_idx(1,2), 1)]
    v = [(b_idx(2,3), 1)]
    w = [(c_idx(1,3), 1)]
    add_term(u, v, w, None)
    
    # Term 33: A12 * B24
    u = [(a_idx(1,2), 1)]
    v = [(b_idx(2,4), 1)]
    w = [(c_idx(1,4), 1)]
    add_term(u, v, w, None)

    # 6. C_R_bot (1x2) = A21 * B_R_top + A22 * B_R_bot
    # A21 * B_R_top: 1x2 * 2x2. Rank 4.
    # A21: A[2,0], A[2,1].
    # B_R_top: B[0,3], B[0,4], B[1,3], B[1,4].
    
    # Term 34: A20 * B03
    u = [(a_idx(2,0), 1)]
    v = [(b_idx(0,3), 1)]
    w = [(c_idx(2,3), 1)]
    add_term(u, v, w, None)
    
    # Term 35: A20 * B13
    u = [(a_idx(2,0), 1)]
    v = [(b_idx(1,3), 1)]
    w = [(c_idx(2,3), 1)]
    add_term(u, v, w, None)
    
    # Term 36: A21 * B04
    u = [(a_idx(2,1), 1)]
    v = [(b_idx(0,4), 1)]
    w = [(c_idx(2,4), 1)]
    add_term(u, v, w, None)
    
    # Term 37: A21 * B14
    u = [(a_idx(2,1), 1)]
    v = [(b_idx(1,4), 1)]
    w = [(c_idx(2,4), 1)]
    add_term(u, v, w, None)
    
    # A22 * B_R_bot: 1x1 * 1x2. Rank 2.
    # A22: A[2,2].
    # B_R_bot: B[2,3], B[2,4].
    
    # Term 38: A22 * B23
    u = [(a_idx(2,2), 1)]
    v = [(b_idx(2,3), 1)]
    w = [(c_idx(2,3), 1)]
    add_term(u, v, w, None)
    
    # Term 39: A22 * B24
    u = [(a_idx(2,2), 1)]
    v = [(b_idx(2,4), 1)]
    w = [(c_idx(2,4), 1)]
    add_term(u, v, w, None)

    return {"U": U, "V": V, "W": W}
