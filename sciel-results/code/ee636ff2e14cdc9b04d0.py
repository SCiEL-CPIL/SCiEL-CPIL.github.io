import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition using block logic.
    # B is split into B1 (3x2), B2 (3x2), B3 (3x1).
    # C is split into C1 (2x2), C2 (2x2), C3 (2x1).
    
    # Helper to add a block decomposition to the global U, V, W
    # block_n, block_m, block_p: dimensions of the sub-problem
    # offset_i, offset_j, offset_k: offsets in the global A, B, C matrices
    # U_block, V_block, W_block: factor matrices for the sub-problem
    
    # We need to accumulate into global U, V, W.
    # Since we don't know the total rank in advance easily, we can build lists of tuples
    # (u_vec, v_vec, w_vec) and then stack them.
    
    terms = []
    
    def add_term(u_idx, v_idx, w_idx):
        # u_idx: index in A (0..nm-1)
        # v_idx: index in B (0..mp-1)
        # w_idx: index in C (0..pn-1)
        terms.append((u_idx, v_idx, w_idx))
        
    # --- Block 1: A (2x3) * B1 (3x2) -> C1 (2x2) ---
    # B1 corresponds to B[:, 0:2]
    # Split A into A_left (2x2) and A_right (2x1)
    # Split B1 into B1_top (2x2) and B1_bottom (1x2)
    
    # Sub-block 1.1: A_left (2x2) * B1_top (2x2) -> C1_part (2x2)
    # Use Strassen for 2x2x2
    # Strassen terms for 2x2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Mapping indices for A_left (cols 0,1 of A)
    # A indices: i in [0,1], j in [0,1]
    # A11: (0,0) -> idx 0
    # A12: (0,1) -> idx 1
    # A21: (1,0) -> idx 3
    # A22: (1,1) -> idx 4
    
    # Mapping indices for B1_top (rows 0,1 of B1, which are rows 0,1 of B)
    # B indices: j in [0,1], k in [0,1] (k is global col 0,1)
    # B11: (0,0) -> idx 0
    # B12: (0,1) -> idx 1
    # B21: (1,0) -> idx 5
    # B22: (1,1) -> idx 6
    
    # Mapping indices for C1 (cols 0,1 of C)
    # C indices: i in [0,1], k in [0,1]
    # C11: (0,0) -> idx 0
    # C12: (0,1) -> idx 1
    # C21: (1,0) -> idx 2
    # C22: (1,1) -> idx 3
    
    # We need to add 7 terms.
    # Each term is a linear combination of A and B entries.
    # The factor matrices U, V, W will have rows corresponding to these linear combinations.
    
    # Instead of manually coding Strassen, I will use a generic "add_rank1" approach
    # but for Strassen, the "rank 1" terms are not simple Kronecker products of basis vectors.
    # They are linear combinations.
    # So U, V, W will contain 0s, 1s, -1s.
    
    # Let's define the Strassen vectors.
    # U vectors (for A):
    # M1: A11 + A22 -> [1, 0, 0, 1] (in 2x2 space)
    # M2: A21 + A22 -> [0, 0, 1, 1]
    # M3: A11 -> [1, 0, 0, 0]
    # M4: A22 -> [0, 0, 0, 1]
    # M5: A11 + A12 -> [1, 1, 0, 0]
    # M6: A21 - A11 -> [-1, 0, 1, 0]
    # M7: A12 - A22 -> [0, 1, 0, -1]
    
    # V vectors (for B):
    # M1: B11 + B22 -> [1, 0, 0, 1]
    # M2: B11 -> [1, 0, 0, 0]
    # M3: B12 - B22 -> [0, 1, 0, -1]
    # M4: B21 - B11 -> [-1, 0, 1, 0]
    # M5: B22 -> [0, 0, 0, 1]
    # M6: B11 + B12 -> [1, 1, 0, 0]
    # M7: B21 + B22 -> [0, 0, 1, 1]
    
    # W vectors (for C):
    # M1: C11 + C22 -> [1, 0, 0, 1]
    # M2: -C21 -> [0, 0, -1, 0]
    # M3: C12 -> [0, 1, 0, 0]
    # M4: C11 + C21 -> [1, 0, 1, 0]
    # M5: -C11 + C22 -> [-1, 0, 0, 1]
    # M6: C22 -> [0, 0, 0, 1]
    # M7: C11 -> [1, 0, 0, 0]
    
    # Wait, the standard Strassen identity is:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So W vectors should be:
    # M1: C11 + C22 -> [1, 0, 0, 1]
    # M2: C21 - C22 -> [0, 0, 1, -1]  (Note: C22 has -M2)
    # M3: C12 + C22 -> [0, 1, 0, 1]
    # M4: C11 + C21 -> [1, 0, 1, 0]
    # M5: -C11 + C12 -> [-1, 1, 0, 0]
    # M6: C22 -> [0, 0, 0, 1]
    # M7: C11 -> [1, 0, 0, 0]
    
    # I will implement a helper to add a 2x2x2 Strassen block.
    
    def add_strassen_block(a_offset, b_offset, c_offset):
        # a_offset: index of A[0,0]
        # b_offset: index of B[0,0]
        # c_offset: index of C[0,0]
        
        # A indices:
        # A11: a_offset
        # A12: a_offset + 1
        # A21: a_offset + 3
        # A22: a_offset + 4
        
        # B indices:
        # B11: b_offset
        # B12: b_offset + 1
        # B21: b_offset + 5
        # B22: b_offset + 6
        
        # C indices:
        # C11: c_offset
        # C12: c_offset + 1
        # C21: c_offset + 2
        # C22: c_offset + 3
        
        # We add 7 terms.
        # Each term is defined by U_vec, V_vec, W_vec.
        # We store them as (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
        
        # M1
        u_idx = [a_offset, a_offset+4]
        u_val = [1, 1]
        v_idx = [b_offset, b_offset+6]
        v_val = [1, 1]
        w_idx = [c_offset, c_offset+3]
        w_val = [1, 1]
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
        
        # M2
        u_idx = [a_offset+3, a_offset+4]
        u_val = [1, 1]
        v_idx = [b_offset]
        v_val = [1]
        w_idx = [c_offset+2, c_offset+3]
        w_val = [1, -1]
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
        
        # M3
        u_idx = [a_offset]
        u_val = [1]
        v_idx = [b_offset+1, b_offset+6]
        v_val = [1, -1]
        w_idx = [c_offset+1, c_offset+3]
        w_val = [1, 1]
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
        
        # M4
        u_idx = [a_offset+4]
        u_val = [1]
        v_idx = [b_offset+5, b_offset]
        v_val = [1, -1]
        w_idx = [c_offset, c_offset+2]
        w_val = [1, 1]
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
        
        # M5
        u_idx = [a_offset, a_offset+1]
        u_val = [1, 1]
        v_idx = [b_offset+6]
        v_val = [1]
        w_idx = [c_offset, c_offset+1]
        w_val = [-1, 1]
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
        
        # M6
        u_idx = [a_offset+3, a_offset]
        u_val = [1, -1]
        v_idx = [b_offset, b_offset+1]
        v_val = [1, 1]
        w_idx = [c_offset+3]
        w_val = [1]
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
        
        # M7
        u_idx = [a_offset+1, a_offset+4]
        u_val = [1, -1]
        v_idx = [b_offset+5, b_offset+6]
        v_val = [1, 1]
        w_idx = [c_offset]
        w_val = [1]
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
        
    def add_trivial_2x2x1_block(a_offset, b_offset, c_offset):
        # A is 2x2, B is 2x1, C is 2x1
        # A11, A12, A21, A22
        # B1, B2
        # C1, C2
        # C1 = A11*B1 + A12*B2
        # C2 = A21*B1 + A22*B2
        # 4 terms
        # T1: A11 * B1 -> C1
        # T2: A12 * B2 -> C1
        # T3: A21 * B1 -> C2
        # T4: A22 * B2 -> C2
        
        # A indices: a_offset, a_offset+1, a_offset+3, a_offset+4
        # B indices: b_offset, b_offset+2 (since B is 2x1, step is 1? No, B is 2x1, so B[0,0], B[1,0]. Indices 0, 1)
        # Wait, B is 2x1. m=2, p=1. Indices: j*p+k.
        # B[0,0] -> 0. B[1,0] -> 1.
        # C indices: c_offset, c_offset+1 (C is 2x1. i*p+k? No, C is n x p. 2x1. Indices i*p+k. 0, 1)
        
        # T1
        terms.append(([a_offset], [1], [b_offset], [1], [c_offset], [1]))
        # T2
        terms.append(([a_offset+1], [1], [b_offset+1], [1], [c_offset], [1]))
        # T3
        terms.append(([a_offset+3], [1], [b_offset], [1], [c_offset+1], [1]))
        # T4
        terms.append(([a_offset+4], [1], [b_offset+1], [1], [c_offset+1], [1]))
        
    def add_trivial_2x1x1_block(a_offset, b_offset, c_offset):
        # A is 2x1, B is 1x1, C is 2x1
        # A1, A2
        # B1
        # C1, C2
        # C1 = A1*B1
        # C2 = A2*B1
        # 2 terms
        
        # A indices: a_offset, a_offset+2 (A is 2x1. m=1. Indices 0, 1)
        # B indices: b_offset
        # C indices: c_offset, c_offset+1
        
        terms.append(([a_offset], [1], [b_offset], [1], [c_offset], [1]))
        terms.append(([a_offset+1], [1], [b_offset], [1], [c_offset+1], [1]))
        
    # --- Block 1: A (2x3) * B1 (3x2) ---
    # A_left (2x2): A[:, 0:2]
    # A_right (2x1): A[:, 2:3]
    # B1_top (2x2): B[0:2, 0:2]
    # B1_bottom (1x2): B[2:3, 0:2]
    
    # 1.1: A_left * B1_top (2x2x2)
    # A_left offset: 0
    # B1_top offset: 0
    # C1 offset: 0
    add_strassen_block(0, 0, 0)
    
    # 1.2: A_right * B1_bottom (2x1x2)
    # A_right offset: A[:, 2] -> indices 2, 5 (i*m+j. i=0,j=2 -> 2. i=1,j=2 -> 5)
    # B1_bottom offset: B[2, 0:2] -> indices 2*5+0=10, 2*5+1=11
    # C1 offset: 0 (same C1)
    # Wait, 2x1x2 multiplication.
    # A is 2x1. B is 1x2. C is 2x2.
    # C11 = A1 B1
    # C12 = A1 B2
    # C21 = A2 B1
    # C22 = A2 B2
    # 4 terms.
    # A indices: 2, 5
    # B indices: 10, 11
    # C indices: 0, 1, 2, 3
    
    terms.append(([2], [1], [10], [1], [0], [1]))
    terms.append(([2], [1], [11], [1], [1], [1]))
    terms.append(([5], [1], [10], [1], [2], [1]))
    terms.append(([5], [1], [11], [1], [3], [1]))
    
    # --- Block 2: A (2x3) * B2 (3x2) ---
    # B2 is B[:, 2:4]
    # B2_top: B[0:2, 2:4] -> indices 0*5+2=2, 0*5+3=3, 1*5+2=7, 1*5+3=8
    # B2_bottom: B[2:3, 2:4] -> indices 2*5+2=12, 2*5+3=13
    # C2 is C[:, 2:4] -> indices 2, 3, 4, 5
    
    # 2.1: A_left * B2_top (2x2x2)
    # A_left offset: 0
    # B2_top offset: 2
    # C2 offset: 2
    add_strassen_block(0, 2, 2)
    
    # 2.2: A_right * B2_bottom (2x1x2)
    # A_right offset: 2
    # B2_bottom offset: 12
    # C2 offset: 2
    terms.append(([2], [1], [12], [1], [2], [1]))
    terms.append(([2], [1], [13], [1], [3], [1]))
    terms.append(([5], [1], [12], [1], [4], [1]))
    terms.append(([5], [1], [13], [1], [5], [1]))
    
    # --- Block 3: A (2x3) * B3 (3x1) ---
    # B3 is B[:, 4:5]
    # B3_top: B[0:2, 4:5] -> indices 4, 9
    # B3_bottom: B[2:3, 4:5] -> index 14
    # C3 is C[:, 4:5] -> indices 6, 7
    
    # 3.1: A_left * B3_top (2x2x1)
    # A_left offset: 0
    # B3_top offset: 4
    # C3 offset: 6
    add_trivial_2x2x1_block(0, 4, 6)
    
    # 3.2: A_right * B3_bottom (2x1x1)
    # A_right offset: 2
    # B3_bottom offset: 14
    # C3 offset: 6
    add_trivial_2x1x1_block(2, 14, 6)
    
    # Now, convert terms to U, V, W matrices.
    rank = len(terms)
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for idx, val in zip(u_idx, u_val):
            U[idx, r] = val
        for idx, val in zip(v_idx, v_val):
            V[idx, r] = val
        for idx, val in zip(w_idx, w_val):
            W[idx, r] = val
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
