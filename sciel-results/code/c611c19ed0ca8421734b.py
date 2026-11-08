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
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize factor matrices
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((16, rank), dtype=np.int64)
    W = np.zeros((16, rank), dtype=np.int64)
    
    # Index helpers
    def u_idx(i, j): return i * 4 + j
    def v_idx(i, j): return i * 4 + j
    def w_idx(i, j): return j * 4 + i # Trace convention: W[k*n+i, r] -> W[col*n + row]
    
    # --- Level 1: 4x4 -> 7 block products (2x2 matrices) ---
    # Strassen terms for 4x4 matrices split into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B01-B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    
    # Define linear combinations for A and B blocks.
    # Each block is 2x2. We represent linear combinations as sparse vectors or explicit index lists.
    # For efficiency and clarity in the recursive step, we define the terms explicitly.
    
    # Indices for A blocks (row-major within block, then global)
    # A00: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    # A01: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
    # A10: (2,0), (2,1), (3,0), (3,1) -> 8, 9, 12, 13
    # A11: (2,2), (2,3), (3,2), (3,3) -> 10, 11, 14, 15
    
    # Indices for B blocks
    # B00: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    # B01: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
    # B10: (2,0), (2,1), (3,0), (3,1) -> 8, 9, 12, 13
    # B11: (2,2), (2,3), (3,2), (3,3) -> 10, 11, 14, 15
    
    # We define the 7 block-level products as (L_combo, R_combo) where L_combo and R_combo 
    # are lists of (global_index, coefficient) pairs for the 2x2 matrix elements.
    # However, it's easier to define the 2x2 matrix coefficients directly for the sub-Strassen.
    
    # Let's define the structure of each of the 7 block products.
    # Each block product is a multiplication of two 2x2 matrices, say X * Y.
    # X and Y are linear combinations of the 4 blocks of A and B.
    # We will apply Strassen to X * Y.
    
    # Strassen for 2x2 matrices X, Y:
    # M0 = (X00+X11)(Y00+Y11)
    # M1 = (X10+X11)Y00
    # M2 = X00(Y01-Y11)
    # M3 = X11(Y10-Y00)
    # M4 = (X00+X01)Y11
    # M5 = (X10-X00)(Y00+Y01)
    # M6 = (X01-X11)(Y10+Y11)
    
    # Result Z = X*Y:
    # Z00 = M0 + M3 - M4 + M6
    # Z01 = M2 + M4
    # Z10 = M1 + M3
    # Z11 = M0 - M1 + M2 + M5
    
    # We need to map the contributions of Z00, Z01, Z10, Z11 to the global 4x4 result C.
    # Block t contributes to specific 2x2 blocks of C with specific coefficients.
    
    # Block Maps: term_index -> [(c_block_row, c_block_col, coeff)]
    # P0 (t=0): C00 (1), C11 (1)
    # P1 (t=1): C10 (1), C11 (-1)
    # P2 (t=2): C01 (1), C11 (1)
    # P3 (t=3): C00 (1), C10 (1)
    # P4 (t=4): C00 (-1), C01 (1)
    # P5 (t=5): C11 (1)
    # P6 (t=6): C00 (1)
    
    block_maps = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    # Define X and Y for each term t.
    # X and Y are 2x2 matrices. Each element is a linear combination of A and B elements.
    # We represent an element as a list of (global_idx, coeff).
    
    # Helper to create a single element combo
    def ce(idx, coeff=1):
        return [(idx, coeff)]
        
    # Helper to add combos
    def add_c(c1, c2):
        d = {}
        for i, v in c1: d[i] = d.get(i, 0) + v
        for i, v in c2: d[i] = d.get(i, 0) + v
        return [(k, v) for k, v in d.items() if v != 0]
        
    # Helper to subtract combos
    def sub_c(c1, c2):
        d = {}
        for i, v in c1: d[i] = d.get(i, 0) + v
        for i, v in c2: d[i] = d.get(i, 0) - v
        return [(k, v) for k, v in d.items() if v != 0]
        
    # A Block Elements (Global Indices)
    # A00: [0, 1, 4, 5]
    # A01: [2, 3, 6, 7]
    # A10: [8, 9, 12, 13]
    # A11: [10, 11, 14, 15]
    A00 = [[(0,1)], [(1,1)], [(4,1)], [(5,1)]] # 2x2 list of combos
    A01 = [[(2,1)], [(3,1)], [(6,1)], [(7,1)]]
    A10 = [[(8,1)], [(9,1)], [(12,1)], [(13,1)]]
    A11 = [[(10,1)], [(11,1)], [(14,1)], [(15,1)]]
    
    # B Block Elements
    B00 = [[(0,1)], [(1,1)], [(4,1)], [(5,1)]]
    B01 = [[(2,1)], [(3,1)], [(6,1)], [(7,1)]]
    B10 = [[(8,1)], [(9,1)], [(12,1)], [(13,1)]]
    B11 = [[(10,1)], [(11,1)], [(14,1)], [(15,1)]]
    
    # Construct X and Y for each term t
    # X, Y are 2x2 matrices (lists of 4 combos)
    # Order: 00, 01, 10, 11
    
    def block_add(M1, M2):
        return [add_c(M1[i], M2[i]) for i in range(4)]
        
    def block_sub(M1, M2):
        return [sub_c(M1[i], M2[i]) for i in range(4)]
        
    # T0: X = A00+A11, Y = B00+B11
    X0 = block_add(A00, A11)
    Y0 = block_add(B00, B11)
    
    # T1: X = A10+A11, Y = B00
    X1 = block_add(A10, A11)
    Y1 = B00
    
    # T2: X = A00, Y = B01-B11
    X2 = A00
    Y2 = block_sub(B01, B11)
    
    # T3: X = A11, Y = B10-B00
    X3 = A11
    Y3 = block_sub(B10, B00)
    
    # T4: X = A00+A01, Y = B11
    X4 = block_add(A00, A01)
    Y4 = B11
    
    # T5: X = A10-A00, Y = B00+B01
    X5 = block_sub(A10, A00)
    Y5 = block_add(B00, B01)
    
    # T6: X = A01-A11, Y = B10+B11
    X6 = block_sub(A01, A11)
    Y6 = block_add(B10, B11)
    
    X_list = [X0, X1, X2, X3, X4, X5, X6]
    Y_list = [Y0, Y1, Y2, Y3, Y4, Y5, Y6]
    
    # Sub-contributions of Mk to Z (local 2x2 result of X*Y)
    # Z00 = M0 + M3 - M4 + M6
    # Z01 = M2 + M4
    # Z10 = M1 + M3
    # Z11 = M0 - M1 + M2 + M5
    # Indices: 0->Z00, 1->Z01, 2->Z10, 3->Z11
    
    sub_contribs = [
        [(0, 1), (3, 1)],  # M0
        [(2, 1), (3, -1)], # M1
        [(1, 1), (3, 1)],  # M2
        [(0, 1), (2, 1)],  # M3
        [(0, -1), (1, 1)], # M4
        [(3, 1)],          # M5
        [(0, 1)]           # M6
    ]
    
    # Fill U, V, W
    for t in range(7):
        X = X_list[t]
        Y = Y_list[t]
        
        # Extract X, Y elements
        x00, x01, x10, x11 = X[0], X[1], X[2], X[3]
        y00, y01, y10, y11 = Y[0], Y[1], Y[2], Y[3]
        
        # Compute 7 Strassen products M0..M6
        # M0: (x00+x11)(y00+y11)
        m0_l = add_c(x00, x11)
        m0_r = add_c(y00, y11)
        
        # M1: (x10+x11)y00
        m1_l = add_c(x10, x11)
        m1_r = y00
        
        # M2: x00(y01-y11)
        m2_l = x00
        m2_r = sub_c(y01, y11)
        
        # M3: x11(y10-y00)
        m3_l = x11
        m3_r = sub_c(y10, y00)
        
        # M4: (x00+x01)y11
        m4_l = add_c(x00, x01)
        m4_r = y11
        
        # M5: (x10-x00)(y00+y01)
        m5_l = sub_c(x10, x00)
        m5_r = add_c(y00, y01)
        
        # M6: (x01-x11)(y10+y11)
        m6_l = sub_c(x01, x11)
        m6_r = add_c(y10, y11)
        
        ms = [
            (m0_l, m0_r),
            (m1_l, m1_r),
            (m2_l, m2_r),
            (m3_l, m3_r),
            (m4_l, m4_r),
            (m5_l, m5_r),
            (m6_l, m6_r)
        ]
        
        for k in range(7):
            rank_idx = t * 7 + k
            l_combo, r_combo = ms[k]
            
            # Update U and V
            for idx, coeff in l_combo:
                U[idx, rank_idx] += coeff
            for idx, coeff in r_combo:
                V[idx, rank_idx] += coeff
                
            # Update W
            # For each contribution of Mk to local Z elements
            for z_idx, coeff in sub_contribs[k]:
                # z_idx 0->Z00, 1->Z01, 2->Z10, 3->Z11
                # Local row, col
                lr = z_idx // 2
                lc = z_idx % 2
                
                # Map to global C blocks using block_maps[t]
                for c_br, c_bc, c_coeff in block_maps[t]:
                    # Global row = 2*c_br + lr
                    # Global col = 2*c_bc + lc
                    gr = 2 * c_br + lr
                    gc = 2 * c_bc + lc
                    w_i = w_idx(gr, gc)
                    W[w_i, rank_idx] += coeff * c_coeff
                    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
