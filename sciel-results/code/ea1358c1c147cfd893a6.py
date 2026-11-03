import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # u, v, w are vectors of length 4
    # A = (a,b,c,d), B = (e,f,g,h), C = (c11, c12, c21, c22)
    S_u = [
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Decomposition for 4x4x4 using Strassen
    # Blocks: 11 (0,0), 12 (0,1), 21 (1,0), 22 (1,1)
    # Indices in 4x4 matrix:
    # A11: i in 0,1; j in 0,1
    # A12: i in 0,1; j in 2,3
    # A21: i in 2,3; j in 0,1
    # A22: i in 2,3; j in 2,3
    
    # Strassen formulas for 4x4x4
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # We generate the rank-1 terms for each P_m
    # Each P_m is a product of two 2x2 matrices, decomposed into 7 terms
    
    def get_block_indices(block_id):
        # block_id: 11, 12, 21, 22
        # Returns list of (i, j) indices in 4x4
        if block_id == 11:
            return [(0,0), (0,1), (1,0), (1,1)]
        elif block_id == 12:
            return [(0,2), (0,3), (1,2), (1,3)]
        elif block_id == 21:
            return [(2,0), (2,1), (3,0), (3,1)]
        elif block_id == 22:
            return [(2,2), (2,3), (3,2), (3,3)]
            
    # Coefficients for the 7 P_m terms
    # P1: A11+A22, B11+B22
    # P2: A21+A22, B11
    # P3: A11, B12-B22
    # P4: A22, B21-B11
    # P5: A11+A12, B22
    # P6: A21-A11, B11+B12
    # P7: A12-A22, B21+B22
    
    # We define the linear combinations for A and B blocks
    # For each P_m, we have a set of (coeff, block) pairs for A and B
    
    A_combos = [
        [(1, 11), (1, 22)],
        [(1, 21), (1, 22)],
        [(1, 11)],
        [(1, 22)],
        [(1, 11), (1, 12)],
        [(1, 21), (-1, 11)],
        [(1, 12), (-1, 22)]
    ]
    
    B_combos = [
        [(1, 11), (1, 22)],
        [(1, 11)],
        [(1, 12), (-1, 22)],
        [(1, 21), (-1, 11)],
        [(1, 22)],
        [(1, 11), (1, 12)],
        [(1, 21), (1, 22)]
    ]
    
    # Output block contributions
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # For each output block (11, 12, 21, 22), we sum the contributions from P_m
    # C_block = sum (coeff * P_m)
    # Each P_m is decomposed into 7 rank-1 terms (u_t, v_t, w_t)
    # w_t is a vector of length 4 (for the 2x2 block)
    # We need to map w_t to the 4x4 C indices
    
    def map_w_to_4x4(w_vec, block_id):
        # w_vec is length 4: (c11, c12, c21, c22) of the 2x2 block
        # Returns list of (coeff, i, k) for 4x4 C
        indices = get_block_indices(block_id)
        res = []
        for idx, val in enumerate(w_vec):
            if val != 0:
                i, k = indices[idx]
                res.append((val, i, k))
        return res

    U_rows = []
    V_rows = []
    W_rows = []
    
    # Add terms for M_4,4,4 (k=0..3)
    for m in range(7):
        # Get A and B linear combinations
        a_terms = A_combos[m]
        b_terms = B_combos[m]
        
        # Expand A linear combination to 4x4 indices
        # u_vec is length 16? No, we need to form the u vector for the 2x2 Strassen term
        # The 2x2 Strassen term t has u_t (length 4).
        # We need to construct the full u vector for the 4x4 A.
        
        for t in range(7):
            u_2x2 = S_u[t]
            v_2x2 = S_v[t]
            w_2x2 = S_w[t]
            
            # Construct u_4x4 (length 16)
            u_4x4 = [0] * 16
            for coeff, block_id in a_terms:
                indices = get_block_indices(block_id)
                for idx, val in enumerate(u_2x2):
                    if val != 0:
                        i, j = indices[idx]
                        u_4x4[i * 4 + j] += coeff * val
                        
            # Construct v_4x4 (length 36, but only k=0..3 are used)
            v_4x4 = [0] * 36
            for coeff, block_id in b_terms:
                indices = get_block_indices(block_id)
                for idx, val in enumerate(v_2x2):
                    if val != 0:
                        j, k = indices[idx]
                        v_4x4[j * 9 + k] += coeff * val
                        
            # Construct w_4x4 (length 36, but only k=0..3 are used)
            w_4x4 = [0] * 36
            # Determine which output block this P_m contributes to
            # We need to sum contributions to each output block
            # This is handled by adding to the final W matrix
            
            # For now, store the term
            U_rows.append(u_4x4)
            V_rows.append(v_4x4)
            
            # We need to accumulate W contributions
            # w_2x2 is the vector for the 2x2 result block
            # We need to know which output block (11, 12, 21, 22) this P_m affects
            
    # This approach is getting complicated to handle the W accumulation.
    # Let's just use the schoolbook for M_4,4,4 to be safe and fast, 
    # and see if we can beat 144.
    # Schoolbook M_4,4,4 is rank 16? No, 64.
    # 64 + 64 + 16 = 144.
    # So we MUST use Strassen.
    
    # Let's restart the construction with a clearer W accumulation.
    
    U = [[0] * 114 for _ in range(16)]
    V = [[0] * 114 for _ in range(36)]
    W = [[0] * 114 for _ in range(36)]
    
    col = 0
    
    # M_4,4,4 part 1 (k=0..3)
    # We use the 49 Strassen terms
    for m in range(7):
        a_terms = A_combos[m]
        b_terms = B_combos[m]
        
        for t in range(7):
            u_2x2 = S_u[t]
            v_2x2 = S_v[t]
            w_2x2 = S_w[t]
            
            # Build U column
            for r in range(16): U[r][col] = 0
            for coeff, block_id in a_terms:
                indices = get_block_indices(block_id)
                for idx, val in enumerate(u_2x2):
                    if val != 0:
                        i, j = indices[idx]
                        U[i * 4 + j][col] += coeff * val
                        
            # Build V column
            for r in range(36): V[r][col] = 0
            for coeff, block_id in b_terms:
                indices = get_block_indices(block_id)
                for idx, val in enumerate(v_2x2):
                    if val != 0:
                        j, k = indices[idx]
                        V[j * 9 + k][col] += coeff * val
                        
            # Build W column
            for r in range(36): W[r][col] = 0
            # Determine output block contributions
            # P1 -> C11 (+1), C22 (+1)
            # P2 -> C21 (+1), C22 (-1)
            # P3 -> C12 (+1), C22 (+1)
            # P4 -> C11 (+1), C21 (+1)
            # P5 -> C11 (-1), C12 (+1)
            # P6 -> C22 (+1)
            # P7 -> C11 (+1)
            
            if m == 0: # P1
                w_blocks = [(1, 11), (1, 22)]
            elif m == 1: # P2
                w_blocks = [(1, 21), (-1, 22)]
            elif m == 2: # P3
                w_blocks = [(1, 12), (1, 22)]
            elif m == 3: # P4
                w_blocks = [(1, 11), (1, 21)]
            elif m == 4: # P5
                w_blocks = [(-1, 11), (1, 12)]
            elif m == 5: # P6
                w_blocks = [(1, 22)]
            elif m == 6: # P7
                w_blocks = [(1, 11)]
            
            for coeff, block_id in w_blocks:
                indices = get_block_indices(block_id)
                for idx, val in enumerate(w_2x2):
                    if val != 0:
                        i, k = indices[idx]
                        W[k * 4 + i][col] += coeff * val
                        
            col += 1
            
    # M_4,4,4 part 2 (k=4..7)
    # Same structure, but B indices are shifted by 4
    for m in range(7):
        a_terms = A_combos[m]
        b_terms = B_combos[m]
        
        for t in range(7):
            u_2x2 = S_u[t]
            v_2x2 = S_v[t]
            w_2x2 = S_w[t]
            
            for r in range(16): U[r][col] = 0
            for coeff, block_id in a_terms:
                indices = get_block_indices(block_id)
                for idx, val in enumerate(u_2x2):
                    if val != 0:
                        i, j = indices[idx]
                        U[i * 4 + j][col] += coeff * val
                        
            for r in range(36): V[r][col] = 0
            for coeff, block_id in b_terms:
                indices = get_block_indices(block_id)
                for idx, val in enumerate(v_2x2):
                    if val != 0:
                        j, k = indices[idx]
                        V[j * 9 + (k + 4)][col] += coeff * val
                        
            for r in range(36): W[r][col] = 0
            if m == 0:
                w_blocks = [(1, 11), (1, 22)]
            elif m == 1:
                w_blocks = [(1, 21), (-1, 22)]
            elif m == 2:
                w_blocks = [(1, 12), (1, 22)]
            elif m == 3:
                w_blocks = [(1, 11), (1, 21)]
            elif m == 4:
                w_blocks = [(-1, 11), (1, 12)]
            elif m == 5:
                w_blocks = [(1, 22)]
            elif m == 6:
                w_blocks = [(1, 11)]
                
            for coeff, block_id in w_blocks:
                indices = get_block_indices(block_id)
                for idx, val in enumerate(w_2x2):
                    if val != 0:
                        i, k = indices[idx]
                        W[(k + 4) * 4 + i][col] += coeff * val
                        
            col += 1
            
    # M_4,4,1 part (k=8)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 9 + 8][col] = 1
            W[8 * 4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
