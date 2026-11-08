import numpy as np
import time

def construct():
    # Deadline guard: 150s limit, use 140s to be safe
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 8
    
    # We aim for rank 98 (2 * 49)
    # This is achieved by splitting the 4x8 B matrix into two 4x4 blocks
    # and performing two independent 4x4x4 Strassen multiplications.
    
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 base matrices (rank 7)
    # These are 4x7 matrices representing the coefficients for A, B, C blocks
    # Indexing: row = block_row * 2 + block_col, col = strassen_step
    
    # Standard Strassen coefficients for 2x2 multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # U2: Coefficients for A blocks
    # V2: Coefficients for B blocks
    # W2: Coefficients for C blocks
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen P0..P6
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B01-B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    
    # A coefficients (U2)
    # col 0: P0 -> A00(1), A11(1)
    U2[0, 0] = 1; U2[3, 0] = 1
    # col 1: P1 -> A10(1), A11(1)
    U2[2, 1] = 1; U2[3, 1] = 1
    # col 2: P2 -> A00(1)
    U2[0, 2] = 1
    # col 3: P3 -> A11(1)
    U2[3, 3] = 1
    # col 4: P4 -> A00(1), A01(1)
    U2[0, 4] = 1; U2[1, 4] = 1
    # col 5: P5 -> A10(1), A00(-1)
    U2[2, 5] = 1; U2[0, 5] = -1
    # col 6: P6 -> A01(1), A11(-1)
    U2[1, 6] = 1; U2[3, 6] = -1
    
    # B coefficients (V2)
    # col 0: P0 -> B00(1), B11(1)
    V2[0, 0] = 1; V2[3, 0] = 1
    # col 1: P1 -> B00(1)
    V2[0, 1] = 1
    # col 2: P2 -> B01(1), B11(-1)
    V2[1, 2] = 1; V2[3, 2] = -1
    # col 3: P3 -> B10(1), B00(-1)
    V2[2, 3] = 1; V2[0, 3] = -1
    # col 4: P4 -> B11(1)
    V2[3, 4] = 1
    # col 5: P5 -> B00(1), B01(1)
    V2[0, 5] = 1; V2[1, 5] = 1
    # col 6: P6 -> B10(1), B11(1)
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # C coefficients (W2)
    # col 0: P0 -> C00(1), C11(1)
    W2[0, 0] = 1; W2[3, 0] = 1
    # col 1: P1 -> C10(1), C11(-1)
    W2[2, 1] = 1; W2[3, 1] = -1
    # col 2: P2 -> C01(1), C11(1)
    W2[1, 2] = 1; W2[3, 2] = 1
    # col 3: P3 -> C00(1), C10(1)
    W2[0, 3] = 1; W2[2, 3] = 1
    # col 4: P4 -> C00(-1), C01(1)
    W2[0, 4] = -1; W2[1, 4] = 1
    # col 5: P5 -> C11(1)
    W2[3, 5] = 1
    # col 6: P6 -> C00(1)
    W2[0, 6] = 1
    
    # Build the 4x4x4 decomposition (rank 49)
    # We will do this twice for the two halves of the 4x8 matrix.
    
    # Helper to build one 4x4x4 block decomposition into U, V, W at a specific column offset
    def build_4x4_block(offset, b_offset):
        # U is shared for both blocks if we structure it correctly?
        # No, in the tensor decomposition, U, V, W are global.
        # For the first block (B cols 0-3), V indices are 0-31.
        # For the second block (B cols 4-7), V indices are 32-63.
        # U indices are 0-15.
        # W indices:
        #   Block 0: C cols 0-3 -> W indices 0-15 (since W[k*n+i, r], k is col of C)
        #   Block 1: C cols 4-7 -> W indices 16-31
        
        for p_k in range(7):
            for s_m in range(7):
                col = offset + p_k * 7 + s_m
                
                # U contribution
                # U2[s_m, p_k] is the coefficient for A block
                # A block index from U2 row: r*2+c
                # We need to map this to global A index
                # The Strassen decomposition is recursive.
                # The 4x4 decomposition is built from 2x2 Strassen.
                # The "p_k" here refers to the top-level Strassen step for the 4x4 matrix.
                # Wait, the code above defines U2 as 4x7, which corresponds to the 2x2 Strassen.
                # To build a 4x4 decomposition, we need to apply this to each 2x2 block of A?
                # No, the standard way is to use the 4x4 Strassen directly or recursively.
                # The previous code used a recursive-like expansion.
                
                # Let's stick to the previous logic which worked, but refactor it.
                # The previous logic iterated over 7 P_k and 7 S_m.
                # This implies it was treating the 4x4 as a 2x2 of 2x2 blocks.
                # P_k corresponds to the 7 products of the 2x2 Strassen applied to the 2x2 blocks.
                # S_m corresponds to the 7 products of the 2x2 Strassen applied to the inner 2x2 blocks.
                
                # Let's re-verify the previous code's logic.
                # It had A_specs, B_specs, C_specs for the 7 P_k.
                # And it iterated over m_s (7) for the inner Strassen.
                
                # I will replicate that structure but cleaner.
                
                pass # Placeholder, actual logic below
    
    # Re-implementing the block building logic from the original but with distinct variable names
    # A_specs: list of 7 lists, each containing (coeff, block_idx) for A
    # block_idx 0=A00, 1=A01, 2=A10, 3=A11
    
    A_specs = [
        [(1, 0), (1, 3)], # P0
        [(1, 2), (1, 3)], # P1
        [(1, 0)],         # P2
        [(1, 3)],         # P3
        [(1, 0), (1, 1)], # P4
        [(1, 2), (-1, 0)],# P5
        [(1, 1), (-1, 3)],# P6
    ]
    
    B_specs = [
        [(1, 0), (1, 3)], # P0
        [(1, 0)],         # P1
        [(1, 1), (-1, 3)],# P2
        [(1, 2), (-1, 0)],# P3
        [(1, 3)],         # P4
        [(1, 0), (1, 1)], # P5
        [(1, 2), (1, 3)], # P6
    ]
    
    C_specs = [
        [(1, 0), (1, 3)], # P0
        [(1, 2), (-1, 3)], # P1
        [(1, 1), (1, 3)], # P2
        [(1, 0), (1, 2)], # P3
        [(-1, 0), (1, 1)], # P4
        [(1, 3)], # P5
        [(1, 0)], # P6
    ]
    
    # U is independent of the B-block half, so we can fill it once and it applies to all 98 cols?
    # No, the tensor decomposition is a sum of rank-1 terms.
    # U_r, V_r, W_r.
    # If we have two independent multiplications, we have two sets of rank-1 terms.
    # The U part is the same for both?
    # Yes, if we structure the decomposition such that the A-contraction is shared.
    # However, in the tensor form, each rank-1 term is a full tri-linear form.
    # If we use the same U for both halves, we must ensure the V and W are disjoint or structured.
    # The previous code did:
    # U[global_a_idx, col] += val
    # U[global_a_idx, col + 49] += val
    # This implies U is identical for both halves.
    
    # Let's fill U for the first 49 cols, then copy to the next 49.
    # And fill V and W separately for each half.
    
    for p_k in range(7):
        for s_m in range(7):
            col1 = p_k * 7 + s_m
            col2 = 49 + p_k * 7 + s_m
            
            # U update
            for coeff_a, block_idx in A_specs[p_k]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_r = bi * 2 + r
                        global_c = bj * 2 + c
                        global_a_idx = global_r * 4 + global_c
                        a_eff_idx = r * 2 + c
                        val = coeff_a * U2[a_eff_idx, s_m]
                        if val != 0:
                            U[global_a_idx, col1] += val
                            U[global_a_idx, col2] += val
            
            # V and W for Half 0 (B cols 0-3)
            # V index: row = global_r * 8 + global_c (where global_c is 0-3)
            # W index: row = global_k * 4 + global_i (where global_k is 0-3)
            
            for coeff_b, block_idx in B_specs[p_k]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_r = bi * 2 + r
                        global_c = bj * 2 + c
                        global_b_idx = global_r * 8 + global_c
                        b_eff_idx = r * 2 + c
                        val = coeff_b * V2[b_eff_idx, s_m]
                        if val != 0:
                            V[global_b_idx, col1] += val
            
            for coeff_c, c_block_idx in C_specs[p_k]:
                bi, bj = divmod(c_block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_i = bi * 2 + r
                        global_k = bj * 2 + c
                        global_w_idx = global_k * 4 + global_i
                        p_eff_idx = r * 2 + c
                        val = coeff_c * W2[p_eff_idx, s_m]
                        if val != 0:
                            W[global_w_idx, col1] += val
            
            # V and W for Half 1 (B cols 4-7)
            # V index: row = global_r * 8 + (global_c + 4)
            # W index: row = (global_k + 4) * 4 + global_i
            
            for coeff_b, block_idx in B_specs[p_k]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_r = bi * 2 + r
                        global_c = bj * 2 + c
                        global_b_idx = global_r * 8 + (global_c + 4)
                        b_eff_idx = r * 2 + c
                        val = coeff_b * V2[b_eff_idx, s_m]
                        if val != 0:
                            V[global_b_idx, col2] += val
            
            for coeff_c, c_block_idx in C_specs[p_k]:
                bi, bj = divmod(c_block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_i = bi * 2 + r
                        global_k = bj * 2 + c
                        global_w_idx = (global_k + 4) * 4 + global_i
                        p_eff_idx = r * 2 + c
                        val = coeff_c * W2[p_eff_idx, s_m]
                        if val != 0:
                            W[global_w_idx, col2] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
