import numpy as np
import time

def construct():
    n, m, p = 4, 4, 9
    # Rank 114: 49 (cols 0-3) + 49 (cols 4-7) + 16 (col 8)
    rank = 114
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Define 2x2 Strassen decomposition
    # U, V, W are 4x7 matrices
    # Indices 0..3 for 2x2 matrices (row-major)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: (A00+A11)(B00+B11)
    U2[:, 0] = [1, 0, 0, 1]
    V2[:, 0] = [1, 0, 0, 1]
    W2[:, 0] = [1, 0, 0, 1] # C00 += 1, C11 += 1
    
    # Term 1: (A10+A11)B00
    U2[:, 1] = [0, 0, 1, 1]
    V2[:, 1] = [1, 0, 0, 0]
    W2[:, 1] = [0, 0, 1, -1] # C10 += 1, C11 -= 1
    
    # Term 2: A00(B01-B11)
    U2[:, 2] = [1, 0, 0, 0]
    V2[:, 2] = [0, 1, 0, -1]
    W2[:, 2] = [0, 1, 0, 1] # C01 += 1, C11 += 1
    
    # Term 3: A11(B10-B00)
    U2[:, 3] = [0, 0, 0, 1]
    V2[:, 3] = [-1, 0, 1, 0]
    W2[:, 3] = [1, 0, 1, 0] # C00 += 1, C10 += 1
    
    # Term 4: (A00+A01)B11
    U2[:, 4] = [1, 1, 0, 0]
    V2[:, 4] = [0, 0, 0, 1]
    W2[:, 4] = [-1, 1, 0, 0] # C00 -= 1, C01 += 1
    
    # Term 5: (A10-A00)(B00+B01)
    U2[:, 5] = [-1, 0, 1, 0]
    V2[:, 5] = [1, 1, 0, 0]
    W2[:, 5] = [0, 0, 0, 1] # C11 += 1
    
    # Term 6: (A01-A11)(B10+B11)
    U2[:, 6] = [0, 1, 0, -1]
    V2[:, 6] = [0, 0, 1, 1]
    W2[:, 6] = [1, 0, 0, 0] # C00 += 1
    
    def add_4x4_block(col_offset, b_col_offset):
        """
        Adds the 49-term decomposition for M4,4,4.
        b_col_offset: 0 for first 4 cols, 4 for next 4 cols.
        """
        # Block indices in A (16 elements)
        # A00: 0,1,4,5
        # A01: 2,3,6,7
        # A10: 8,9,12,13
        # A11: 10,11,14,15
        
        # Block indices in B (16 elements for 4 cols)
        # B00: 0,1,4,5
        # B01: 2,3,6,7
        # B10: 8,9,12,13
        # B11: 10,11,14,15
        
        # Block indices in C (16 elements for 4 cols)
        # C00: 0,1,4,5
        # C01: 2,3,6,7
        # C10: 8,9,12,13
        # C11: 10,11,14,15
        
        # Strassen terms for 4x4
        # Each term is (X, Y) where X is linear combo of A blocks, Y of B blocks
        # X, Y are 2x2 matrices of block coefficients
        # We represent X as 4 coefficients [x00, x01, x10, x11]
        
        strassen_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1]), # P0
            ([0, 0, 1, 1], [1, 0, 0, 0]), # P1
            ([1, 0, 0, 0], [0, 1, 0, -1]), # P2
            ([0, 0, 0, 1], [-1, 0, 1, 0]), # P3
            ([1, 1, 0, 0], [0, 0, 0, 1]), # P4
            ([-1, 0, 1, 0], [1, 1, 0, 0]), # P5
            ([0, 1, 0, -1], [0, 0, 1, 1])  # P6
        ]
        
        # Block index maps for A, B, C
        # A blocks: 00, 01, 10, 11
        # Indices in A (16):
        A_block_indices = [
            [0, 1, 4, 5], # A00
            [2, 3, 6, 7], # A01
            [8, 9, 12, 13], # A10
            [10, 11, 14, 15] # A11
        ]
        
        # B blocks (relative to b_col_offset)
        # B is 4x9. We are processing 4 columns starting at b_col_offset.
        # B indices in V (36): j*9 + k.
        # For a 4x4 submatrix starting at col b_col_offset:
        # B00 (j in 0,1; k in 0,1): j*9 + (b_col_offset + k)
        B_block_indices = []
        for b_idx in range(4):
            row_block = b_idx // 2
            col_block = b_idx % 2
            # row_block: 0 or 1. col_block: 0 or 1.
            # j in {2*row_block, 2*row_block+1}
            # k in {b_col_offset + 2*col_block, b_col_offset + 2*col_block+1}
            inds = []
            for j in range(2*row_block, 2*row_block+2):
                for k in range(b_col_offset + 2*col_block, b_col_offset + 2*col_block+2):
                    inds.append(j*9 + k)
            B_block_indices.append(inds)
            
        # C blocks (relative to output)
        # C is 4x9. We are processing 4 columns starting at b_col_offset.
        # C indices in W (36): k*4 + i.
        # k in {b_col_offset + 2*col_block, ...}
        # i in {2*row_block, ...}
        C_block_indices = []
        for c_idx in range(4):
            # c_idx 0: C00 (i 0,1; k 0,1)
            # c_idx 1: C01 (i 0,1; k 2,3)
            # c_idx 2: C10 (i 2,3; k 0,1)
            # c_idx 3: C11 (i 2,3; k 2,3)
            row_block = c_idx // 2
            col_block = c_idx % 2
            inds = []
            for k in range(b_col_offset + 2*col_block, b_col_offset + 2*col_block+2):
                for i in range(2*row_block, 2*row_block+2):
                    inds.append(k*4 + i)
            C_block_indices.append(inds)
            
        # Iterate through Strassen terms
        for m in range(7):
            x_coeffs, y_coeffs = strassen_terms[m]
            # x_coeffs: [x00, x01, x10, x11]
            # y_coeffs: [y00, y01, y10, y11]
            
            # For each sub-term in 2x2 decomposition
            for s in range(7):
                u_vec = U2[:, s]
                v_vec = V2[:, s]
                w_vec = W2[:, s]
                
                # Construct global U (16)
                # U_global = sum_{block} u_vec[block_pos] * A_block[block]
                U_global = np.zeros(16)
                for b in range(4):
                    # b: 0->00, 1->01, 2->10, 3->11
                    # u_vec index: b
                    coeff = u_vec[b]
                    if coeff != 0:
                        U_global[A_block_indices[b]] += coeff
                        
                # Construct global V (36)
                V_global = np.zeros(36)
                for b in range(4):
                    coeff = v_vec[b]
                    if coeff != 0:
                        V_global[B_block_indices[b]] += coeff
                        
                # Construct global W (36)
                W_global = np.zeros(36)
                for b in range(4):
                    coeff = w_vec[b]
                    if coeff != 0:
                        W_global[C_block_indices[b]] += coeff
                        
                U[:, col_offset] = U_global
                V[:, col_offset] = V_global
                W[:, col_offset] = W_global
                col_offset += 1
                
        return col_offset

    # Process cols 0-3
    col = add_4x4_block(0, 0)
    
    # Process cols 4-7
    col = add_4x4_block(col, 4)
    
    # Process col 8 (Schoolbook 16 terms)
    for i in range(4):
        for j in range(4):
            U[i*4+j, col] = 1
            V[j*9+8, col] = 1
            W[8*4+i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
