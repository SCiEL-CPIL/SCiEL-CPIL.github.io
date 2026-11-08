import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # U, V, W are 16x49 matrices
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((16, rank), dtype=np.int64)
    W = np.zeros((16, rank), dtype=np.int64)
    
    # Helper to map 2D indices to 1D
    def u_idx(i, j): return i * 4 + j
    def v_idx(i, j): return i * 4 + j
    def w_idx(i, j): return j * 4 + i # Trace convention: W[k*n+i, r] -> W[col*n + row]
    
    # Strassen algorithm for 2x2 matrices
    # Inputs: A (2x2), B (2x2)
    # Outputs: 7 products M1..M7
    # Each product is a scalar multiplication of two linear combinations.
    # We represent linear combinations as lists of (index, coeff)
    
    # We are doing 4x4 multiplication by splitting into 2x2 blocks of 2x2 submatrices.
    # Block indices:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Where P1..P7 are 2x2 matrix products.
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # Each Pi is a 2x2 product, computed via Strassen (7 scalar mults).
    # Total 7 * 7 = 49 scalar mults.
    
    # We need to define the linear combinations for A and B for each of the 7 block products.
    # Then for each block product, apply Strassen to get 7 scalar products.
    
    # Define block element ranges
    # Block (bi, bj) of A covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # Global index for A: u_idx(r, c)
    # Global index for B: v_idx(r, c)
    
    def get_a_indices(bi, bj):
        """Return list of 4 global U indices for block (bi, bj) in row-major order"""
        return [
            u_idx(2*bi, 2*bj),
            u_idx(2*bi, 2*bj+1),
            u_idx(2*bi+1, 2*bj),
            u_idx(2*bi+1, 2*bj+1)
        ]
        
    def get_b_indices(bi, bj):
        """Return list of 4 global V indices for block (bi, bj) in row-major order"""
        return [
            v_idx(2*bi, 2*bj),
            v_idx(2*bi, 2*bj+1),
            v_idx(2*bi+1, 2*bj),
            v_idx(2*bi+1, 2*bj+1)
        ]
    
    # Define the 7 block-level terms as linear combinations of the 16 A and B elements.
    # Each term is a pair (L_comb, R_comb) where L_comb is a list of (u_idx, coeff) and R_comb is (v_idx, coeff).
    # Actually, L and R are 2x2 matrices of linear combinations.
    # Let's represent the 2x2 matrix L as a list of 4 linear combinations (for elements 00, 01, 10, 11).
    
    # Term 0: P1 = (A00 + A11) * (B00 + B11)
    # L0: A00 + A11
    # L0[0] = A00[0] + A11[0]
    # L0[1] = A00[1] + A11[1]
    # L0[2] = A00[2] + A11[2]
    # L0[3] = A00[3] + A11[3]
    
    def add_combos(c1, c2):
        """Add two lists of (idx, coeff)"""
        d = {}
        for idx, c in c1:
            d[idx] = d.get(idx, 0) + c
        for idx, c in c2:
            d[idx] = d.get(idx, 0) + c
        return [(k, v) for k, v in d.items() if v != 0]
        
    def sub_combos(c1, c2):
        """Subtract two lists of (idx, coeff)"""
        d = {}
        for idx, c in c1:
            d[idx] = d.get(idx, 0) + c
        for idx, c in c2:
            d[idx] = d.get(idx, 0) - c
        return [(k, v) for k, v in d.items() if v != 0]
    
    # Pre-calculate indices for blocks
    A00_idx = get_a_indices(0, 0)
    A01_idx = get_a_indices(0, 1)
    A10_idx = get_a_indices(1, 0)
    A11_idx = get_a_indices(1, 1)
    
    B00_idx = get_b_indices(0, 0)
    B01_idx = get_b_indices(0, 1)
    B10_idx = get_b_indices(1, 0)
    B11_idx = get_b_indices(1, 1)
    
    # Convert index lists to combo lists with coeff 1
    def to_combo(indices):
        return [(i, 1) for i in indices]
        
    A00 = to_combo(A00_idx)
    A01 = to_combo(A01_idx)
    A10 = to_combo(A10_idx)
    A11 = to_combo(A11_idx)
    
    B00 = to_combo(B00_idx)
    B01 = to_combo(B01_idx)
    B10 = to_combo(B10_idx)
    B11 = to_combo(B11_idx)
    
    # Define L and R for each of the 7 block terms
    # L and R are lists of 4 combos (for 2x2 elements)
    
    L_terms = []
    R_terms = []
    
    # T0: (A00+A11) * (B00+B11)
    L0 = [
        add_combos(A00[0:1], A11[0:1]), # Element 00
        add_combos(A00[1:2], A11[1:2]), # Element 01
        add_combos(A00[2:3], A11[2:3]), # Element 10
        add_combos(A00[3:4], A11[3:4])  # Element 11
    ]
    R0 = [
        add_combos(B00[0:1], B11[0:1]),
        add_combos(B00[1:2], B11[1:2]),
        add_combos(B00[2:3], B11[2:3]),
        add_combos(B00[3:4], B11[3:4])
    ]
    L_terms.append(L0)
    R_terms.append(R0)
    
    # T1: (A10+A11) * B00
    L1 = [
        add_combos(A10[0:1], A11[0:1]),
        add_combos(A10[1:2], A11[1:2]),
        add_combos(A10[2:3], A11[2:3]),
        add_combos(A10[3:4], A11[3:4])
    ]
    R1 = [
        B00[0:1], B00[1:2], B00[2:3], B00[3:4]
    ]
    L_terms.append(L1)
    R_terms.append(R1)
    
    # T2: A00 * (B01-B11)
    L2 = [
        A00[0:1], A00[1:2], A00[2:3], A00[3:4]
    ]
    R2 = [
        sub_combos(B01[0:1], B11[0:1]),
        sub_combos(B01[1:2], B11[1:2]),
        sub_combos(B01[2:3], B11[2:3]),
        sub_combos(B01[3:4], B11[3:4])
    ]
    L_terms.append(L2)
    R_terms.append(R2)
    
    # T3: A11 * (B10-B00)
    L3 = [
        A11[0:1], A11[1:2], A11[2:3], A11[3:4]
    ]
    R3 = [
        sub_combos(B10[0:1], B00[0:1]),
        sub_combos(B10[1:2], B00[1:2]),
        sub_combos(B10[2:3], B00[2:3]),
        sub_combos(B10[3:4], B00[3:4])
    ]
    L_terms.append(L3)
    R_terms.append(R3)
    
    # T4: (A00+A01) * B11
    L4 = [
        add_combos(A00[0:1], A01[0:1]),
        add_combos(A00[1:2], A01[1:2]),
        add_combos(A00[2:3], A01[2:3]),
        add_combos(A00[3:4], A01[3:4])
    ]
    R4 = [
        B11[0:1], B11[1:2], B11[2:3], B11[3:4]
    ]
    L_terms.append(L4)
    R_terms.append(R4)
    
    # T5: (A10-A00) * (B00+B01)
    L5 = [
        sub_combos(A10[0:1], A00[0:1]),
        sub_combos(A10[1:2], A00[1:2]),
        sub_combos(A10[2:3], A00[2:3]),
        sub_combos(A10[3:4], A00[3:4])
    ]
    R5 = [
        add_combos(B00[0:1], B01[0:1]),
        add_combos(B00[1:2], B01[1:2]),
        add_combos(B00[2:3], B01[2:3]),
        add_combos(B00[3:4], B01[3:4])
    ]
    L_terms.append(L5)
    R_terms.append(R5)
    
    # T6: (A01-A11) * (B10+B11)
    L6 = [
        sub_combos(A01[0:1], A11[0:1]),
        sub_combos(A01[1:2], A11[1:2]),
        sub_combos(A01[2:3], A11[2:3]),
        sub_combos(A01[3:4], A11[3:4])
    ]
    R6 = [
        add_combos(B10[0:1], B11[0:1]),
        add_combos(B10[1:2], B11[1:2]),
        add_combos(B10[2:3], B11[2:3]),
        add_combos(B10[3:4], B11[3:4])
    ]
    L_terms.append(L6)
    R_terms.append(R6)
    
    # Now, for each term t (0-6), we have a 2x2 matrix L and R.
    # We apply Strassen to compute L * R.
    # L = [[l00, l01], [l10, l11]], R = [[r00, r01], [r10, r11]]
    # M1 = (l00+l11)(r00+r11)
    # M2 = (l10+l11)r00
    # M3 = l00(r01-r11)
    # M4 = l11(r10-r00)
    # M5 = (l00+l01)r11
    # M6 = (l10-l00)(r00+r01)
    # M7 = (l01-l11)(r10+r11)
    
    # Result P = L*R:
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    # Each Mk is a scalar product of two linear combinations.
    # We add these to U, V, W.
    
    # Sub-contributions of Mk to the 2x2 result P (local indices 00, 01, 10, 11)
    # M1 -> P00: 1, P11: 1
    # M2 -> P10: 1, P11: -1
    # M3 -> P01: 1, P11: 1
    # M4 -> P00: 1, P10: 1
    # M5 -> P00: -1, P01: 1
    # M6 -> P11: 1
    # M7 -> P00: 1
    
    sub_contrib = [
        [(0, 1), (3, 1)],      # M1: P00, P11
        [(2, 1), (3, -1)],     # M2: P10, P11
        [(1, 1), (3, 1)],      # M3: P01, P11
        [(0, 1), (2, 1)],      # M4: P00, P10
        [(0, -1), (1, 1)],     # M5: P00, P01
        [(3, 1)],              # M6: P11
        [(0, 1)]               # M7: P00
    ]
    
    # Map local P indices (00, 01, 10, 11) to global C blocks
    # Block t contributes to specific global C blocks.
    # T0 (P1) -> C00 (1), C11 (1)
    # T1 (P2) -> C10 (1), C11 (-1)
    # T2 (P3) -> C01 (1), C11 (1)
    # T3 (P4) -> C00 (1), C10 (1)
    # T4 (P5) -> C00 (-1), C01 (1)
    # T5 (P6) -> C11 (1)
    # T6 (P7) -> C00 (1)
    
    # Wait, the block contributions are:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # So for each term t, and each local element (lr, lc) of the 2x2 result,
    # we need to map to global C elements.
    
    # Local (lr, lc) in block t corresponds to:
    # Block 0 (P1): C00. Global C indices for block 00: rows 0-1, cols 0-1.
    # Local (0,0) -> Global (0,0), Local (0,1) -> Global (0,1), etc.
    
    def get_global_c_indices(block_bi, block_bj, lr, lc):
        """
        Returns the global W index for the element at local (lr, lc) within block (block_bi, block_bj).
        W index is w_idx(row, col) = col * 4 + row.
        Global row = 2*block_bi + lr
        Global col = 2*block_bj + lc
        """
        r = 2 * block_bi + lr
        c = 2 * block_bj + lc
        return w_idx(r, c)
        
    # Define block mappings for each term t
    # t -> list of (block_bi, block_bj, coeff)
    # P1 (t=0): C00 (1), C11 (1)
    # P2 (t=1): C10 (1), C11 (-1)
    # P3 (t=2): C01 (1), C11 (1)
    # P4 (t=3): C00 (1), C10 (1)
    # P5 (t=4): C00 (-1), C01 (1)
    # P6 (t=5): C11 (1)
    # P7 (t=6): C00 (1)
    
    block_maps = [
        [(0, 0, 1), (1, 1, 1)],  # t=0
        [(1, 0, 1), (1, 1, -1)], # t=1
        [(0, 1, 1), (1, 1, 1)],  # t=2
        [(0, 0, 1), (1, 0, 1)],  # t=3
        [(0, 0, -1), (0, 1, 1)], # t=4
        [(1, 1, 1)],             # t=5
        [(0, 0, 1)]              # t=6
    ]
    
    for t in range(7):
        L = L_terms[t]
        R = R_terms[t]
        
        # Extract L and R elements
        l00, l01, l10, l11 = L[0], L[1], L[2], L[3]
        r00, r01, r10, r11 = R[0], R[1], R[2], R[3]
        
        # Generate 7 sub-products
        # M1: (l00+l11) * (r00+r11)
        x1 = add_combos(l00, l11)
        y1 = add_combos(r00, r11)
        
        # M2: (l10+l11) * r00
        x2 = add_combos(l10, l11)
        y2 = r00
        
        # M3: l00 * (r01-r11)
        x3 = l00
        y3 = sub_combos(r01, r11)
        
        # M4: l11 * (r10-r00)
        x4 = l11
        y4 = sub_combos(r10, r00)
        
        # M5: (l00+l01) * r11
        x5 = add_combos(l00, l01)
        y5 = r11
        
        # M6: (l10-l00) * (r00+r01)
        x6 = sub_combos(l10, l00)
        y6 = add_combos(r00, r01)
        
        # M7: (l01-l11) * (r10+r11)
        x7 = sub_combos(l01, l11)
        y7 = add_combos(r10, r11)
        
        sub_prods = [
            (x1, y1),
            (x2, y2),
            (x3, y3),
            (x4, y4),
            (x5, y5),
            (x6, y6),
            (x7, y7)
        ]
        
        for k in range(7):
            rank_idx = t * 7 + k
            x_k, y_k = sub_prods[k]
            
            # Fill U and V
            for idx, coeff in x_k:
                U[idx, rank_idx] += coeff
            for idx, coeff in y_k:
                V[idx, rank_idx] += coeff
                
            # Fill W
            # For each local contribution of Mk to P (local 2x2)
            for (local_idx, sub_coeff) in sub_contrib[k]:
                lr = local_idx // 2
                lc = local_idx % 2
                
                # Map local (lr, lc) to global C via block_maps[t]
                for (bi, bj, block_coeff) in block_maps[t]:
                    # The local element (lr, lc) of the 2x2 result in block (bi, bj)
                    # corresponds to global C element at (2*bi+lr, 2*bj+lc)
                    w_idx_val = get_global_c_indices(bi, bj, lr, lc)
                    W[w_idx_val, rank_idx] += sub_coeff * block_coeff
    
    # Convert to lists of lists as required by interface?
    # The interface says: factor matrices of shapes (n*m, rank)...
    # The current program returns lists of lists. I will return lists of lists to be safe.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
