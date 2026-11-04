import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # Strassen 2x2 matrices
    # U, V, W are 2x2 matrices
    strassen_U = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    strassen_V = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    strassen_W = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]

    # Part 1: 4x4x4 using Kronecker product of two Strassen decompositions
    # Rank = 7 * 7 = 49
    rank_44 = 49
    # Part 2: 4x4x3 using schoolbook
    # Rank = 4 * 4 * 3 = 48
    rank_43 = 48
    total_rank = rank_44 + rank_43
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to add Kronecker product of 2x2 matrices to U, V, W
    def add_kron_term(U_mat, V_mat, W_mat, r_idx, offset_U, offset_V, offset_W, p_stride):
        # U_mat, V_mat, W_mat are 2x2
        # Kronecker product gives 4x4
        # We need to map 4x4 indices to flattened indices
        # U: row i*m + j. Here m=4. So i*4 + j.
        # V: row j*p + k. Here p=7. So j*7 + k.
        # W: row k*n + i. Here n=4. So k*4 + i.
        
        # Compute Kronecker product
        # A (2x2) kron B (2x2) -> C (4x4)
        # C[2*i + j, 2*k + l] = A[i, k] * B[j, l]
        
        # We iterate over all entries of the resulting 4x4 matrices
        # U_kron[2*i + j, 2*k + l] = U_mat[i, k] * U_mat_inner[j, l]?
        # Wait, the Kronecker product is (Outer U) ⊗ (Inner U).
        # Let Outer be O, Inner be I.
        # (O ⊗ I)[2*i + j, 2*k + l] = O[i, k] * I[j, l]
        
        # Here, the "Outer" indices correspond to the block structure (2x2 blocks of 2x2 matrices).
        # The "Inner" indices correspond to the elements within the 2x2 blocks.
        
        # So U_flat index is (2*i + j) * 4 + (2*k + l) ?
        # No, U is n x m = 4 x 4.
        # Row index is i_row * m + j_col = i_row * 4 + j_col.
        # i_row = 2*i + j, j_col = 2*k + l.
        # So U_flat index = (2*i + j) * 4 + (2*k + l).
        
        # Similarly for V:
        # V is m x p = 4 x 7.
        # We are filling the first 4 columns (k=0..3).
        # Row index is j_row * p + k_col = j_row * 7 + k_col.
        # j_row = 2*j + l ? No.
        # Let's trace the indices carefully.
        # A is 4x4. Blocks A_00, A_01, A_10, A_11.
        # A_00 is rows 0,1 cols 0,1.
        # A_01 is rows 0,1 cols 2,3.
        # A_10 is rows 2,3 cols 0,1.
        # A_11 is rows 2,3 cols 2,3.
        # So block (I, J) contains elements with row 2*I + i, col 2*J + j.
        
        # U corresponds to A.
        # U[2*I + i, 2*J + j] = O[I, J] * I_mat[i, j].
        # Flat index: (2*I + i) * 4 + (2*J + j).
        
        # V corresponds to B.
        # B is 4x7. We use first 4 cols.
        # B_00 is rows 0,1 cols 0,1.
        # B_01 is rows 0,1 cols 2,3.
        # B_10 is rows 2,3 cols 0,1.
        # B_11 is rows 2,3 cols 2,3.
        # V[2*I + i, 2*J + j] = O[I, J] * I_mat[i, j].
        # Flat index: (2*I + i) * 7 + (2*J + j).
        
        # W corresponds to C.
        # C is 4x7. We use first 4 cols.
        # C_00 is rows 0,1 cols 0,1.
        # C_01 is rows 0,1 cols 2,3.
        # C_10 is rows 2,3 cols 0,1.
        # C_11 is rows 2,3 cols 2,3.
        # W[2*I + i, 2*J + j] = O[I, J] * I_mat[i, j].
        # Flat index: (2*J + j) * 4 + (2*I + i).
        # Note: W row index is k*n + i.
        # k is the column of C (which is row of W in block sense? No).
        # C[k, i].
        # Block (J, I) of C corresponds to rows 2*J + j, cols 2*I + i.
        # So W flat index = (2*J + j) * 4 + (2*I + i).
        
        for I in range(2):
            for J in range(2):
                o_coeff = O_mat[I][J]
                if o_coeff == 0: continue
                for i in range(2):
                    for j in range(2):
                        i_coeff = I_mat[i][j]
                        val = o_coeff * i_coeff
                        if val == 0: continue
                        
                        # U
                        u_idx = (2*I + i) * 4 + (2*J + j)
                        U[u_idx][r_idx] += val
                        
                        # V
                        v_idx = (2*I + i) * 7 + (2*J + j)
                        V[v_idx][r_idx] += val
                        
                        # W
                        w_idx = (2*J + j) * 4 + (2*I + i)
                        W[w_idx][r_idx] += val

    # We need to pass the Outer and Inner matrices to the helper.
    # Let's modify the helper to take O and I matrices directly.
    
    def add_kron_term_full(O_U, I_U, O_V, I_V, O_W, I_W, r_idx):
        for I in range(2):
            for J in range(2):
                ou = O_U[I][J]
                ov = O_V[I][J]
                ow = O_W[I][J]
                if ou == 0 and ov == 0 and ow == 0: continue
                for i in range(2):
                    for j in range(2):
                        iu = I_U[i][j]
                        iv = I_V[i][j]
                        iw = I_W[i][j]
                        
                        u_val = ou * iu
                        v_val = ov * iv
                        w_val = ow * iw
                        
                        if u_val != 0:
                            u_idx = (2*I + i) * 4 + (2*J + j)
                            U[u_idx][r_idx] += u_val
                        if v_val != 0:
                            v_idx = (2*I + i) * 7 + (2*J + j)
                            V[v_idx][r_idx] += v_val
                        if w_val != 0:
                            w_idx = (2*J + j) * 4 + (2*I + i)
                            W[w_idx][r_idx] += w_val

    for s in range(7):
        for r in range(7):
            r_idx = current_rank
            current_rank += 1
            add_kron_term_full(strassen_U[s], strassen_U[r], 
                               strassen_V[s], strassen_V[r], 
                               strassen_W[s], strassen_W[r], r_idx)
            
    # Part 2: 4x4x3 Schoolbook
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                r_idx = current_rank
                current_rank += 1
                k = k_prime + 4
                
                u_idx = i * 4 + j
                v_idx = j * 7 + k
                w_idx = k * 4 + i
                
                U[u_idx][r_idx] = 1.0
                V[v_idx][r_idx] = 1.0
                W[w_idx][r_idx] = 1.0
                
    return {"U": U, "V": V, "W": W}
