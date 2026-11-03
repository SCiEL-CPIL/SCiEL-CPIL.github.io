import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    rank = 65
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    # Inner Strassen 2x2 factors
    # u, v, w are length 4
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    inner_u = []
    inner_v = []
    inner_w = []
    
    # M1 = (A00+A11)(B00+B11) -> C00+C11
    # M1 contributes to C00 (idx 0) and C11 (idx 3)
    # Inner factors for M1?
    # The inner decomposition computes X*Y.
    # X = A00+A11, Y = B00+B11.
    # The factors for X*Y are the same as any 2x2 multiplication.
    # We just need the 7 factors for 2x2 multiplication.
    
    # Let's define the 7 factors for 2x2 multiplication explicitly.
    # T_2 = sum_r u_r v_r w_r
    # u_r, v_r, w_r in R^4
    
    # Standard Strassen factors (flattened row-major)
    # Note: The factors are not unique. We need a valid set.
    # I will use the standard ones.
    
    # Helper to create vector
    def vec(d): return np.array(d)
    
    # 1. (A00+A11)(B00+B11)
    # u: [1, 0, 0, 1]
    # v: [1, 0, 0, 1]
    # w: [1, 0, 0, 1] (Contributes to C00 and C11)
    inner_u.append(vec([1, 0, 0, 1]))
    inner_v.append(vec([1, 0, 0, 1]))
    inner_w.append(vec([1, 0, 0, 1]))
    
    # 2. (A10+A11)B00
    # u: [0, 0, 1, 1]
    # v: [1, 0, 0, 0]
    # w: [0, 0, 1, 0] (Contributes to C10)
    inner_u.append(vec([0, 0, 1, 1]))
    inner_v.append(vec([1, 0, 0, 0]))
    inner_w.append(vec([0, 0, 1, 0]))
    
    # 3. A00(B01-B11)
    # u: [1, 0, 0, 0]
    # v: [0, 1, 0, -1]
    # w: [0, 1, 0, 0] (Contributes to C01)
    inner_u.append(vec([1, 0, 0, 0]))
    inner_v.append(vec([0, 1, 0, -1]))
    inner_w.append(vec([0, 1, 0, 0]))
    
    # 4. A11(B10-B00)
    # u: [0, 0, 0, 1]
    # v: [0, 0, 1, -1]
    # w: [1, 0, 1, 0] (Contributes to C00 and C10)
    inner_u.append(vec([0, 0, 0, 1]))
    inner_v.append(vec([0, 0, 1, -1]))
    inner_w.append(vec([1, 0, 1, 0]))
    
    # 5. (A00+A01)B11
    # u: [1, 1, 0, 0]
    # v: [0, 0, 0, 1]
    # w: [0, 1, 0, 1] (Contributes to C01 and C11)
    inner_u.append(vec([1, 1, 0, 0]))
    inner_v.append(vec([0, 0, 0, 1]))
    inner_w.append(vec([0, 1, 0, 1]))
    
    # 6. (A10-A00)(B00+B01)
    # u: [-1, 0, 1, 0]
    # v: [1, 1, 0, 0]
    # w: [0, 0, 1, 0] (Contributes to C10)
    inner_u.append(vec([-1, 0, 1, 0]))
    inner_v.append(vec([1, 1, 0, 0]))
    inner_w.append(vec([0, 0, 1, 0]))
    
    # 7. (A01-A11)(B10+B11)
    # u: [0, 1, 0, -1]
    # v: [0, 0, 1, 1]
    # w: [-1, 0, 0, 1] (Contributes to C00 and C11)
    inner_u.append(vec([0, 1, 0, -1]))
    inner_v.append(vec([0, 0, 1, 1]))
    inner_w.append(vec([-1, 0, 0, 1]))
    
    # Outer Strassen Linear Forms
    # A blocks: 00 (idx 0-3), 01 (4-7), 10 (8-11), 11 (12-15)
    # B blocks: same
    # C blocks: same
    
    outer_defs = [
        # k, A_blocks, B_blocks, C_blocks
        # (coeff, block_idx)
        {
            'A': [(1, 0), (1, 3)], 'B': [(1, 0), (1, 3)], 'C': [(1, 0), (1, 3)]
        },
        {
            'A': [(1, 2), (1, 3)], 'B': [(1, 0)], 'C': [(1, 2)]
        },
        {
            'A': [(1, 0)], 'B': [(1, 1), (-1, 3)], 'C': [(1, 1)]
        },
        {
            'A': [(1, 3)], 'B': [(1, 2), (-1, 0)], 'C': [(1, 0), (1, 2)]
        },
        {
            'A': [(1, 0), (1, 1)], 'B': [(1, 3)], 'C': [(1, 1), (1, 3)]
        },
        {
            'A': [(-1, 0), (1, 2)], 'B': [(1, 0), (1, 1)], 'C': [(1, 2)]
        },
        {
            'A': [(1, 1), (-1, 3)], 'B': [(1, 2), (1, 3)], 'C': [(-1, 0), (1, 3)]
        }
    ]
    
    col = 0
    for k in range(7):
        def_A = outer_defs[k]['A']
        def_B = outer_defs[k]['B']
        def_C = outer_defs[k]['C']
        
        for r in range(7):
            u_full = np.zeros(16)
            v_full = np.zeros(20)
            w_full = np.zeros(20)
            
            # Fill U
            for coeff, block_idx in def_A:
                # Block idx 0 -> 0:4, 1 -> 4:8, 2 -> 8:12, 3 -> 12:16
                start = block_idx * 4
                u_full[start:start+4] += coeff * inner_u[r]
                
            # Fill V
            for coeff, block_idx in def_B:
                start = block_idx * 4
                # V indices for block j: j*5 + k
                # Wait, V is 20xR.
                # B is 4x5.
                # Block 0 is rows 0-1, cols 0-3.
                # Block 1 is rows 0-1, col 4.
                # Block 2 is rows 2-3, cols 0-3.
                # Block 3 is rows 2-3, col 4.
                
                # My block definition for 4x4 assumed 2x2 blocks of 2x2.
                # Block 0: A[0:2, 0:2]. Indices 0,1,4,5 in flattened 4x4?
                # Flattened A (4x4):
                # 0 1 2 3
                # 4 5 6 7
                # 8 9 10 11
                # 12 13 14 15
                # Block 0 (00): 0,1,4,5.
                # Block 1 (01): 2,3,6,7.
                # Block 2 (10): 8,9,12,13.
                # Block 3 (11): 10,11,14,15.
                
                # My inner_u assumed row-major 2x2: 0,1,2,3 -> 00,01,10,11.
                # So inner_u[0] is A00, inner_u[1] is A01, inner_u[2] is A10, inner_u[3] is A11.
                # In flattened 4x4:
                # A00 is at 0.
                # A01 is at 1.
                # A10 is at 4.
                # A11 is at 5.
                
                # So mapping inner idx -> full idx:
                # 0 -> 0
                # 1 -> 1
                # 2 -> 4
                # 3 -> 5
                # For Block 0.
                
                # For Block 1 (cols 2,3):
                # A00 (of block) -> 2
                # A01 -> 3
                # A10 -> 6
                # A11 -> 7
                
                # Mapping function:
                def get_block_indices(block_idx):
                    if block_idx == 0: return [0, 1, 4, 5]
                    if block_idx == 1: return [2, 3, 6, 7]
                    if block_idx == 2: return [8, 9, 12, 13]
                    if block_idx == 3: return [10, 11, 14, 15]
                
                # Apply to V
                # V is 4x5.
                # Block 0: rows 0-1, cols 0-3.
                # Flattened V (20):
                # Row 0: 0 1 2 3 4
                # Row 1: 5 6 7 8 9
                # Row 2: 10 11 12 13 14
                # Row 3: 15 16 17 18 19
                
                # Block 0 in V:
                # B00 (of block) -> 0
                # B01 -> 1
                # B10 -> 5
                # B11 -> 6
                
                # Block 1 in V (cols 2,3):
                # B00 -> 2
                # B01 -> 3
                # B10 -> 7
                # B11 -> 8
                
                # Block 2 (rows 2,3):
                # B00 -> 10
                # B01 -> 11
                # B10 -> 15
                # B11 -> 16
                
                # Block 3:
                # B00 -> 12
                # B01 -> 13
                # B10 -> 17
                # B11 -> 18
                
                def get_v_block_indices(block_idx):
                    if block_idx == 0: return [0, 1, 5, 6]
                    if block_idx == 1: return [2, 3, 7, 8]
                    if block_idx == 2: return [10, 11, 15, 16]
                    if block_idx == 3: return [12, 13, 17, 18]
                
                v_indices = get_v_block_indices(block_idx)
                for i in range(4):
                    v_full[v_indices[i]] += coeff * inner_v[r][i]
            
            # Fill W
            # W is 4x5 output C.
            # C blocks are 2x2.
            # Block 0: rows 0-1, cols 0-3.
            # Indices in W (20):
            # 0,1,4,5 (Block 0)
            # 2,3,6,7 (Block 1)
            # 8,9,12,13 (Block 2)
            # 10,11,14,15 (Block 3)
            
            def get_w_block_indices(block_idx):
                if block_idx == 0: return [0, 1, 4, 5]
                if block_idx == 1: return [2, 3, 6, 7]
                if block_idx == 2: return [8, 9, 12, 13]
                if block_idx == 3: return [10, 11, 14, 15]
            
            for coeff, block_idx in def_C:
                w_indices = get_w_block_indices(block_idx)
                for i in range(4):
                    w_full[w_indices[i]] += coeff * inner_w[r][i]
            
            U[:, col] = u_full
            V[:, col] = v_full
            W[:, col] = w_full
            col += 1
            
    # Add last column terms
    for i in range(4):
        for j in range(4):
            U[i*4+j, col] = 1
            V[j*5+4, col] = 1
            W[16+i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
