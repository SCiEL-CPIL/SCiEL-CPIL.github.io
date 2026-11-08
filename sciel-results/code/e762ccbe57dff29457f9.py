import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 7 Strassen products for 4x4
    # Each product is defined by linear combos of A blocks and B blocks
    # A blocks: 0=A11, 1=A12, 2=A21, 3=A22
    # B blocks: 0=B11, 1=B12, 2=B21, 3=B22
    
    # Pk = Xk * Yk
    # Xk is sum of c * A_block
    # Yk is sum of c * B_block
    
    products = [
        # P1: (A11+A22)(B11+B22)
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2: (A21+A22)B11
        ({2: 1, 3: 1}, {0: 1}),
        # P3: A11(B12-B22)
        ({0: 1}, {1: 1, 3: -1}),
        # P4: A22(B21-B11)
        ({3: 1}, {2: 1, 0: -1}),
        # P5: (A11+A12)B22
        ({0: 1, 1: 1}, {3: 1}),
        # P6: (A21-A11)(B11+B12)
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        # P7: (A12-A22)(B21+B22)
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    # 7 terms for 2x2 Strassen
    # Each term is (x_coeffs, y_coeffs, z_coeffs)
    # x_coeffs: dict (r,c) -> coeff for X
    # y_coeffs: dict (r,c) -> coeff for Y
    # z_coeffs: dict (r,c) -> coeff for Z (local result)
    
    t2x2 = [
        # P1 term
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): -1}),
        # P2 term
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # P3 term
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        # P4 term
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1}),
        # P5 term
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # P6 term
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # P7 term
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]
    
    # Mapping of Pk to C blocks
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Block indices: 0=C11, 1=C12, 2=C21, 3=C22
    # Coefficients for each Pk in each block
    # p_k -> {block_idx: coeff}
    p_to_c = [
        {0: 1, 3: 1}, # P1 -> C11(1), C22(1)
        {2: 1, 3: -1}, # P2 -> C21(1), C22(-1)
        {1: 1, 3: 1}, # P3 -> C12(1), C22(1)
        {0: 1, 2: 1}, # P4 -> C11(1), C21(1)
        {0: -1, 1: 1}, # P5 -> C11(-1), C12(1)
        {3: 1}, # P6 -> C22(1)
        {0: 1}, # P7 -> C11(1)
    ]
    
    # Helper to get global indices for A block
    def get_a_indices(block):
        # block 0: 0:2, 0:2
        # block 1: 0:2, 2:4
        # block 2: 2:4, 0:2
        # block 3: 2:4, 2:4
        if block == 0:
            return [(0,0), (0,1), (1,0), (1,1)]
        elif block == 1:
            return [(0,2), (0,3), (1,2), (1,3)]
        elif block == 2:
            return [(2,0), (2,1), (3,0), (3,1)]
        elif block == 3:
            return [(2,2), (2,3), (3,2), (3,3)]

    # Helper to get global indices for B block (first half)
    def get_b_indices(block, half):
        # half 0: cols 0-3, half 1: cols 4-7
        # block 0: 0:2, 0:2
        # block 1: 0:2, 2:4
        # block 2: 2:4, 0:2
        # block 3: 2:4, 2:4
        col_offset = 4 * half
        if block == 0:
            return [(0, col_offset+0), (0, col_offset+1), (1, col_offset+0), (1, col_offset+1)]
        elif block == 1:
            return [(0, col_offset+2), (0, col_offset+3), (1, col_offset+2), (1, col_offset+3)]
        elif block == 2:
            return [(2, col_offset+0), (2, col_offset+1), (3, col_offset+0), (3, col_offset+1)]
        elif block == 3:
            return [(2, col_offset+2), (2, col_offset+3), (3, col_offset+2), (3, col_offset+3)]

    # Helper to get global indices for C block
    def get_c_indices(block):
        # block 0: 0:2, 0:2
        # block 1: 0:2, 2:4
        # block 2: 2:4, 0:2
        # block 3: 2:4, 2:4
        if block == 0:
            return [(0,0), (0,1), (1,0), (1,1)]
        elif block == 1:
            return [(0,2), (0,3), (1,2), (1,3)]
        elif block == 2:
            return [(2,0), (2,1), (3,0), (3,1)]
        elif block == 3:
            return [(2,2), (2,3), (3,2), (3,3)]

    col = 0
    for half in range(2):
        for p_idx in range(7):
            x_blocks, y_blocks = products[p_idx]
            for t_idx in range(7):
                x_coeffs, y_coeffs, z_coeffs = t2x2[t_idx]
                
                # Construct U vector
                # U is size 16
                u_vec = np.zeros(16)
                for block, coeff in x_blocks.items():
                    indices = get_a_indices(block)
                    for local_idx, (r, c) in enumerate(indices):
                        # local_idx corresponds to (0,0), (0,1), (1,0), (1,1)
                        # We need to map local (r,c) of the 2x2 block to the coefficient in x_coeffs
                        # x_coeffs is a dict of (lr, lc) -> coeff
                        # The indices list is ordered (0,0), (0,1), (1,0), (1,1)
                        lr, lc = local_idx // 2, local_idx % 2
                        if (lr, lc) in x_coeffs:
                            u_vec[r * m + c] += coeff * x_coeffs[(lr, lc)]
                
                # Construct V vector
                # V is size 32
                v_vec = np.zeros(32)
                for block, coeff in y_blocks.items():
                    indices = get_b_indices(block, half)
                    for local_idx, (r, c) in enumerate(indices):
                        lr, lc = local_idx // 2, local_idx % 2
                        if (lr, lc) in y_coeffs:
                            v_vec[r * p + c] += coeff * y_coeffs[(lr, lc)]
                
                # Construct W vector
                # W is size 32
                w_vec = np.zeros(32)
                # p_to_c[p_idx] gives the blocks and coefficients
                for c_block, c_coeff in p_to_c[p_idx].items():
                    indices = get_c_indices(c_block)
                    for local_idx, (r, c) in enumerate(indices):
                        lr, lc = local_idx // 2, local_idx % 2
                        if (lr, lc) in z_coeffs:
                            # W index is k*n + i. Here r is row of C, c is col of C.
                            # In our notation, C[i,k] or C[r,c]?
                            # Task: W[k*n+i, r].
                            # C is p x n? No, C is n x p.
                            # W index k*n + i.
                            # k is column index of C?
                            # In matrix mult C = A B, C is n x p.
                            # i is row, k is col.
                            # So W index is k*n + i.
                            # Here r is row, c is col.
                            # So W index is c*n + r.
                            w_vec[c * n + r] += c_coeff * z_coeffs[(lr, lc)]
                
                U[:, col] = u_vec
                V[:, col] = v_vec
                W[:, col] = w_vec
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
