import numpy as np
import time

def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 9
    
    # 1. Construct 2x2x2 Strassen vectors
    # u, v, w are length 4
    micro_strassen = [
        (np.array([1,0,0,1]), np.array([1,0,0,1]), np.array([1,0,0,1])),
        (np.array([0,0,1,1]), np.array([1,0,0,0]), np.array([0,1,0,0])),
        (np.array([1,0,0,0]), np.array([0,1,-1,0]), np.array([0,0,1,0])),
        (np.array([0,0,0,1]), np.array([0,-1,1,0]), np.array([-1,0,0,0])),
        (np.array([1,1,0,0]), np.array([0,0,0,1]), np.array([0,0,0,-1])),
        (np.array([1,0,-1,0]), np.array([1,1,0,0]), np.array([1,0,0,0])),
        (np.array([0,1,0,-1]), np.array([0,0,1,1]), np.array([-1,0,1,0]))
    ]
    
    # 2. Construct 4x4x4 decomposition
    # Blocks of A: 11, 12, 21, 22
    # Global indices for A (16 elements)
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # Blocks of B: 11, 12, 21, 22
    # Global indices for B (16 elements)
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    
    # Blocks of C: 11, 12, 21, 22
    # Global indices for C (16 elements)
    # C11: 0,1,4,5
    # C12: 2,3,6,7
    # C21: 8,9,12,13
    # C22: 10,11,14,15
    
    def embed_block(vec, block_idx):
        """Embed a length-4 vector into a length-16 vector at the specified block."""
        res = np.zeros(16)
        if block_idx == 11:
            res[[0,1,4,5]] = vec
        elif block_idx == 12:
            res[[2,3,6,7]] = vec
        elif block_idx == 21:
            res[[8,9,12,13]] = vec
        elif block_idx == 22:
            res[[10,11,14,15]] = vec
        return res

    # Define the 7 macro-terms
    # Each macro term is defined by:
    # - List of (block_idx, coeff) for A
    # - List of (block_idx, coeff) for B
    # - List of (block_idx, coeff) for C
    macros = [
        # M1: (A11+A22)(B11+B22) -> C11 + C22
        ([ (11, 1), (22, 1) ], [ (11, 1), (22, 1) ], [ (11, 1), (22, 1) ]),
        # M2: (A21+A22)B11 -> C12
        ([ (21, 1), (22, 1) ], [ (11, 1) ], [ (12, 1) ]),
        # M3: A11(B12-B22) -> C21
        ([ (11, 1) ], [ (12, 1), (22, -1) ], [ (21, 1) ]),
        # M4: A22(B21-B11) -> -C11
        ([ (22, 1) ], [ (21, 1), (11, -1) ], [ (11, -1) ]),
        # M5: (A11+A12)B22 -> -C22
        ([ (11, 1), (12, 1) ], [ (22, 1) ], [ (22, -1) ]),
        # M6: (A21-A11)(B11+B12) -> C11
        ([ (21, 1), (11, -1) ], [ (11, 1), (12, 1) ], [ (11, 1) ]),
        # M7: (A12-A22)(B21+B22) -> -C11 + C21
        ([ (12, 1), (22, -1) ], [ (21, 1), (22, 1) ], [ (11, -1), (21, 1) ])
    ]

    U_444 = []
    V_444 = []
    W_444 = []
    
    for macro in macros:
        A_blocks, B_blocks, C_blocks = macro
        
        # Construct global A and B vectors for this macro
        u_macro_base = np.zeros(16)
        for block, coeff in A_blocks:
            u_macro_base += coeff * embed_block(np.ones(4), block)
            
        v_macro_base = np.zeros(16)
        for block, coeff in B_blocks:
            v_macro_base += coeff * embed_block(np.ones(4), block)
            
        # For each micro-term in this macro
        for u_micro, v_micro, w_micro in micro_strassen:
            # The micro-term operates on the "combined" blocks.
            # We need to apply the micro-vectors to the specific blocks involved.
            # This is tricky because the micro-vector is a linear combination of the 4 elements of the 2x2 block.
            # But the macro A is a sum of blocks.
            # Actually, the micro-Strassen decomposition is for a 2x2 product.
            # The inputs to the micro-Strassen are the 2x2 matrices formed by the macro blocks.
            # So we just need to "expand" the micro-vectors into the global space.
            
            # u_global for this term:
            # u_micro has 4 values. These correspond to the 4 elements of the 2x2 A_macro.
            # A_macro is formed by A_blocks.
            # If A_macro = A11 + A22, then the "top-left" element of A_macro is A11[0,0] + A22[0,0].
            # So u_global should have u_micro[0] at A11[0,0] and A22[0,0], etc.
            
            u_global = np.zeros(16)
            # We need to map the 4 local indices of the 2x2 block to the global indices of the contributing blocks.
            # Local indices: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
            # Global indices for block 11: 0,1,4,5
            # Global indices for block 22: 10,11,14,15
            
            # We can reuse the embed logic but with scaling from u_micro
            for block, coeff in A_blocks:
                # Create a vector for this block with values from u_micro
                # The order of elements in embed_block is fixed: 0,1,4,5 etc.
                # u_micro is [00, 01, 10, 11]
                # embed_block expects [00, 01, 10, 11]
                u_global += coeff * embed_block(u_micro, block)
                
            v_global = np.zeros(16)
            for block, coeff in B_blocks:
                v_global += coeff * embed_block(v_micro, block)
                
            w_global = np.zeros(16)
            for block, coeff in C_blocks:
                w_global += coeff * embed_block(w_micro, block)
                
            U_444.append(u_global)
            V_444.append(v_global)
            W_444.append(w_global)

    # 3. Construct 4x4x1 Schoolbook decomposition
    # A is 4x4 (16 elements), B is 4x1 (4 elements), C is 4x1 (4 elements)
    U_441 = []
    V_441 = []
    W_441 = []
    
    for i in range(4):
        for j in range(4):
            u = np.zeros(16)
            u[i*4 + j] = 1
            v = np.zeros(4)
            v[j] = 1
            w = np.zeros(4)
            w[i] = 1
            U_441.append(u)
            V_441.append(v)
            W_441.append(w)
            
    # 4. Combine for 4x4x9
    # B is 4x9. Split into B1 (4x4) and B2 (4x5).
    # B2 split into B2a (4x4) and B2b (4x1).
    # C is 4x9. Split into C1 (4x4), C2a (4x4), C2b (4x1).
    
    # Indices for B (36 elements)
    # B1: 0-15
    # B2a: 16-31
    # B2b: 32-35
    
    # Indices for C (36 elements)
    # C1: 0-15
    # C2a: 16-31
    # C2b: 32-35
    
    U_final = []
    V_final = []
    W_final = []
    
    # Add 4x4x4 terms (for B1 -> C1)
    for u, v, w in zip(U_444, V_444, W_444):
        u_full = u # A is 4x4, same for all
        v_full = np.zeros(36)
        v_full[0:16] = v
        w_full = np.zeros(36)
        w_full[0:16] = w
        U_final.append(u_full)
        V_final.append(v_full)
        W_final.append(w_full)
        
    # Add 4x4x4 terms (for B2a -> C2a)
    for u, v, w in zip(U_444, V_444, W_444):
        u_full = u
        v_full = np.zeros(36)
        v_full[16:32] = v
        w_full = np.zeros(36)
        w_full[16:32] = w
        U_final.append(u_full)
        V_final.append(v_full)
        W_final.append(w_full)
        
    # Add 4x4x1 terms (for B2b -> C2b)
    for u, v, w in zip(U_441, V_441, W_441):
        u_full = u
        v_full = np.zeros(36)
        v_full[32:36] = v
        w_full = np.zeros(36)
        w_full[32:36] = w
        U_final.append(u_full)
        V_final.append(v_full)
        W_final.append(w_full)
        
    # Convert to lists
    U_list = [u.tolist() for u in U_final]
    V_list = [v.tolist() for v in V_final]
    W_list = [w.tolist() for w in W_final]
    
    rank = len(U_list)
    
    # Validate
    U_arr = np.array(U_list)
    V_arr = np.array(V_list)
    W_arr = np.array(W_list)
    
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(9):
                s = np.dot(U_arr[i*4+j, :], V_arr[j*9+k, :] * W_arr[k*4+i, :])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    if not valid: break
        
    if not valid:
        # Fallback to schoolbook
        U_sb = np.zeros((16, 144))
        V_sb = np.zeros((36, 144))
        W_sb = np.zeros((36, 144))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_sb[i*4+j, col] = 1
                    V_sb[j*9+k, col] = 1
                    W_sb[k*4+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    return {"U": U_list, "V": V_list, "W": W_list}
