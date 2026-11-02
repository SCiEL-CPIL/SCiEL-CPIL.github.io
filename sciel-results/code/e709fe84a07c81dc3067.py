import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # Rank 97 decomposition:
    # 49 for 4x4x4 (Strassen)
    # 48 for 4x4x3 (Schoolbook)
    
    rank = 97
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    
    # --- Part 1: 4x4x4 Strassen ---
    # Blocks are 2x2.
    # A blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B blocks (for B_L, cols 0-3):
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # C blocks (for C_L, cols 0-3):
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    
    # Strassen steps for 4x4 (using 2x2 blocks)
    # S_k and T_k are 2x2 matrices of coefficients for A and B blocks.
    # We represent S_k as a 4x4 matrix of coefficients for the 4 A blocks?
    # No, S_k is a 2x2 matrix. Each element is a linear combo of A blocks.
    # Actually, S_k IS a 2x2 matrix formed by adding/subtracting A blocks.
    # So S_k[i,j] is either 0, 1, or -1 times an A block.
    
    # Let's define the 7 steps (S, T, C_combination)
    # C_combination tells how M_k contributes to C blocks.
    # M_k = S_k T_k.
    # C_11 += M_1 + M_4 - M_5 + M_7
    # C_12 += M_3 + M_5
    # C_21 += M_2 + M_4
    # C_22 += M_1 - M_2 + M_3 + M_6
    
    steps = [
        # (S_blocks, T_blocks, C_contrib)
        # S_blocks: dict mapping block name to coeff
        # T_blocks: dict mapping block name to coeff
        # C_contrib: dict mapping C block name to coeff
        ({'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1}, {'C11': 1, 'C22': 1}),
        ({'A21': 1, 'A22': 1}, {'B11': 1}, {'C21': 1, 'C22': -1}),
        ({'A11': 1}, {'B12': 1, 'B22': -1}, {'C12': 1, 'C22': 1}),
        ({'A22': 1}, {'B21': 1, 'B11': -1}, {'C11': 1, 'C21': 1}),
        ({'A11': 1, 'A12': 1}, {'B22': 1}, {'C11': -1, 'C12': 1}),
        ({'A21': 1, 'A11': -1}, {'B11': 1, 'B12': 1}, {'C22': 1}),
        ({'A12': 1, 'A22': -1}, {'B21': 1, 'B22': 1}, {'C11': 1}),
    ]
    
    # 2x2x2 Strassen decomposition vectors
    # u, v, w are length 4 vectors corresponding to (11, 12, 21, 22) elements.
    # Order: [00, 01, 10, 11]
    u_22 = [
        [1, 0, 0, 1], # M1: x11+x22
        [0, 0, 1, 1], # M2: x21+x22
        [1, 0, 0, 0], # M3: x11
        [0, 0, 0, 1], # M4: x22
        [1, 1, 0, 0], # M5: x11+x12
        [-1, 0, 1, 0], # M6: x21-x11
        [0, 1, 0, -1], # M7: x12-x22
    ]
    v_22 = [
        [1, 0, 0, 1], # M1: y11+y22
        [1, 0, 0, 0], # M2: y11
        [0, 1, 0, -1], # M3: y12-y22
        [0, 0, 1, -1], # M4: y21-y11
        [0, 0, 0, 1], # M5: y22
        [1, 1, 0, 0], # M6: y11+y12
        [0, 0, 1, 1], # M7: y21+y22
    ]
    w_22 = [
        [1, 0, 0, 1], # M1: z11+1, z22+1
        [0, 0, 1, -1], # M2: z21+1, z22-1
        [0, 1, 0, 1], # M3: z12+1, z22+1
        [1, 0, 1, 0], # M4: z11+1, z21+1
        [-1, 1, 0, 0], # M5: z11-1, z12+1
        [0, 0, 0, 1], # M6: z22+1
        [1, 0, 0, 0], # M7: z11+1
    ]
    
    # Helper to get global indices for a block
    def get_block_indices(block_name, is_A=True):
        # Returns (row_indices, col_indices) for the 2x2 block
        if is_A:
            # A is 4x4
            if block_name == 'A11': return ([0,1], [0,1])
            if block_name == 'A12': return ([0,1], [2,3])
            if block_name == 'A21': return ([2,3], [0,1])
            if block_name == 'A22': return ([2,3], [2,3])
        else:
            # B is 4x4 (for B_L part)
            if block_name == 'B11': return ([0,1], [0,1])
            if block_name == 'B12': return ([0,1], [2,3])
            if block_name == 'B21': return ([2,3], [0,1])
            if block_name == 'B22': return ([2,3], [2,3])
            
    def get_c_block_indices(block_name):
        # C is 4x4 (for C_L part)
        if block_name == 'C11': return ([0,1], [0,1])
        if block_name == 'C12': return ([0,1], [2,3])
        if block_name == 'C21': return ([2,3], [0,1])
        if block_name == 'C22': return ([2,3], [2,3])
        
    # Process Strassen steps
    for step in steps:
        S_blocks, T_blocks, C_contrib = step
        # For each of the 7 rank-1 terms in 2x2x2
        for r in range(7):
            u_vec = u_22[r]
            v_vec = v_22[r]
            w_vec = w_22[r]
            
            # Construct U vector (size 16)
            U_step = np.zeros(16)
            # S_k is formed by linear combination of A blocks.
            # u_vec has 4 elements corresponding to the 4 elements of S_k (2x2).
            # S_k[i,j] = sum_{block} coeff * A_block[i,j]
            # Actually, S_k is a 2x2 matrix.
            # The elements of S_k are linear combinations of A elements.
            # u_vec[r] is the coefficient for the r-th element of S_k in the product.
            # Wait, the 2x2x2 decomposition is:
            # M = sum_r u_r(S) * v_r(T) * w_r(M)
            # Where u_r(S) is a scalar linear form in S.
            # So u_vec[r] is the vector of coefficients for the 4 elements of S.
            # S itself is a linear combination of A blocks.
            # So we need to expand u_vec[r] (which is a vector of 4 coeffs for S's elements)
            # into the 16 elements of A.
            
            # S is a 2x2 matrix.
            # S[0,0] = sum_{b in S_blocks} S_blocks[b] * A_b[0,0]
            # S[0,1] = sum_{b in S_blocks} S_blocks[b] * A_b[0,1]
            # S[1,0] = sum_{b in S_blocks} S_blocks[b] * A_b[1,0]
            # S[1,1] = sum_{b in S_blocks} S_blocks[b] * A_b[1,1]
            
            # u_vec = [u00, u01, u10, u11]
            # The linear form is u00*S[0,0] + u01*S[0,1] + u10*S[1,0] + u11*S[1,1]
            # Substitute S:
            # = u00 * sum_b S_b A_b[0,0] + ...
            # = sum_b S_b (u00 A_b[0,0] + u01 A_b[0,1] + u10 A_b[1,0] + u11 A_b[1,1])
            
            for b_name, coeff in S_blocks.items():
                rows, cols = get_block_indices(b_name, True)
                for i in range(2):
                    for j in range(2):
                        # u_vec index: i*2 + j
                        val = coeff * u_vec[i*2 + j]
                        global_row = rows[i]
                        global_col = cols[j]
                        U_step[global_row * 4 + global_col] = val
                        
            U[:, idx] = U_step
            
            # Construct V vector (size 16)
            V_step = np.zeros(16)
            for b_name, coeff in T_blocks.items():
                rows, cols = get_block_indices(b_name, False)
                for i in range(2):
                    for j in range(2):
                        val = coeff * v_vec[i*2 + j]
                        global_row = rows[i]
                        global_col = cols[j]
                        V_step[global_row * 4 + global_col] = val
                        
            V[:, idx] = V_step
            
            # Construct W vector (size 16 for C_L part)
            # But W is size p*n = 7*4 = 28.
            # C_L corresponds to cols 0-3 of C.
            # So W rows 0-15 (cols 0-3) are affected.
            W_step = np.zeros(28)
            
            # M_k is a 2x2 matrix.
            # w_vec = [w00, w01, w10, w11]
            # Linear form: w00*M[0,0] + w01*M[0,1] + w10*M[1,0] + w11*M[1,1]
            # M is added to C blocks according to C_contrib.
            # C_block += coeff * M
            # So C_block[i,j] += coeff * M[i,j]
            
            for c_name, coeff in C_contrib.items():
                rows, cols = get_c_block_indices(c_name)
                for i in range(2):
                    for j in range(2):
                        val = coeff * w_vec[i*2 + j]
                        global_row = rows[i]
                        global_col = cols[j]
                        # W index: k*n + i ?
                        # Task: W[k*n+i, r].
                        # C is 4x7.
                        # C_L is 4x4.
                        # global_row is row of C (0-3).
                        # global_col is col of C (0-3).
                        # W row index = global_col * 4 + global_row.
                        W_step[global_col * 4 + global_row] = val
                        
            W[:, idx] = W_step
            idx += 1
            
    # --- Part 2: 4x4x3 Schoolbook ---
    # A is 4x4. B_R is 4x3.
    # B_R cols are 4-6 of B.
    # C_R cols are 4-6 of C.
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # A[i,j] * B[j, k+4] -> C[i, k+4]
                U[i * 4 + j, idx] = 1
                V[j * 7 + (k + 4), idx] = 1
                # W index: (k+4)*4 + i
                W[(k + 4) * 4 + i, idx] = 1
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
