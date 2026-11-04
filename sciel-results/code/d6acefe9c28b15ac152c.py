"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Hybrid decomposition:
    # Split p into 4 blocks of size 2.
    # For each block, split n into 2+1.
    # Top 2x4 x 4x2 part: split into two 2x2 x 2x2 Strassen decompositions.
    # Bottom 1x4 x 4x2 part: schoolbook.
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_row(mat, row_idx, val):
        if row_idx < 0 or row_idx >= len(mat):
            return
        mat[row_idx].append(val)

    def add_strassen(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0):
        nonlocal rank
        # a_i0, a_j0: start indices for A (2x2 block)
        # b_j0, b_k0: start indices for B (2x2 block)
        # c_i0, c_k0: start indices for C (2x2 block)
        
        # 7 terms
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # P1 = (A11+A22)(B11+B22) -> C11+1, C22+1
            ([ (a_i0, a_j0, 1), (a_i0+1, a_j0+1, 1) ],
             [ (b_j0, b_k0, 1), (b_j0+1, b_k0+1, 1) ],
             [ (c_i0, c_k0, 1), (c_i0+1, c_k0+1, 1) ]),
             
            # P2 = (A21+A22)B11 -> C21+1, C22+1
            ([ (a_i0+1, a_j0, 1), (a_i0+1, a_j0+1, 1) ],
             [ (b_j0, b_k0, 1) ],
             [ (c_i0+1, c_k0, 1), (c_i0+1, c_k0+1, 1) ]),
             
            # P3 = A11(B21-B22) -> C12+1, C22-1
            ([ (a_i0, a_j0, 1) ],
             [ (b_j0+1, b_k0, 1), (b_j0+1, b_k0+1, -1) ],
             [ (c_i0, c_k0+1, 1), (c_i0+1, c_k0+1, -1) ]),
             
            # P4 = A22(B12-B11) -> C11-1, C21+1
            ([ (a_i0+1, a_j0+1, 1) ],
             [ (b_j0, b_k0, -1), (b_j0, b_k0+1, 1) ],
             [ (c_i0, c_k0, -1), (c_i0+1, c_k0, 1) ]),
             
            # P5 = (A11+A12)B22 -> C12+1, C22+1
            ([ (a_i0, a_j0, 1), (a_i0, a_j0+1, 1) ],
             [ (b_j0+1, b_k0+1, 1) ],
             [ (c_i0, c_k0+1, 1), (c_i0+1, c_k0+1, 1) ]),
             
            # P6 = (A21-A11)(B11+B12) -> C11+1
            ([ (a_i0+1, a_j0, 1), (a_i0, a_j0, -1) ],
             [ (b_j0, b_k0, 1), (b_j0, b_k0+1, 1) ],
             [ (c_i0, c_k0, 1) ]),
             
            # P7 = (A12-A22)(B21+B22) -> C11+1
            ([ (a_i0, a_j0+1, 1), (a_i0+1, a_j0+1, -1) ],
             [ (b_j0+1, b_k0, 1), (b_j0+1, b_k0+1, 1) ],
             [ (c_i0, c_k0, 1) ]),
        ]
        
        for u_list, v_list, w_list in terms:
            # Initialize new column
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for i, j, val in u_list:
                u_col[i * m + j] = val
            for j, k, val in v_list:
                v_col[j * p + k] = val
            for i, k, val in w_list:
                w_col[k * n + i] = val
                
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
            rank += 1

    # 4 blocks in p (size 2 each)
    for r in range(4):
        k_start = r * 2
        
        # Top part: A[0:2, :] * B[:, k_start:k_start+2]
        # Split A[0:2, :] into A[0:2, 0:2] and A[0:2, 2:4]
        # Split B[:, k_start:k_start+2] into B[0:2, k_start:k_start+2] and B[2:4, k_start:k_start+2]
        
        # Strassen 1: A[0:2, 0:2] * B[0:2, k_start:k_start+2]
        add_strassen(0, 0, 0, k_start, 0, k_start)
        
        # Strassen 2: A[0:2, 2:4] * B[2:4, k_start:k_start+2]
        add_strassen(0, 2, 2, k_start, 0, k_start)
        
        # Bottom part: A[2, :] * B[:, k_start:k_start+2]
        # Schoolbook for 1x4 x 4x2
        for j in range(4):
            for k_local in range(2):
                k_global = k_start + k_local
                u_col = [0] * (n * m)
                v_col = [0] * (m * p)
                w_col = [0] * (p * n)
                
                u_col[2 * m + j] = 1
                v_col[j * p + k_global] = 1
                w_col[k_global * n + 2] = 1
                
                U.append(u_col)
                V.append(v_col)
                W.append(w_col)
                rank += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
