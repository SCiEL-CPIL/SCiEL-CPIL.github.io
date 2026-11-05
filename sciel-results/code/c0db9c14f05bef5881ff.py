import numpy as np
import time

DIMENSIONS = [6, 6, 6]
DEADLINE = 135.0  # Safety margin from 150s limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    def get_schoolbook_3x3():
        n3 = 3
        rank3 = 27
        U3 = np.zeros((9, rank3))
        V3 = np.zeros((9, rank3))
        W3 = np.zeros((9, rank3))
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    col = i*9 + j*3 + k
                    U3[i*3+j, col] = 1.0
                    V3[j*3+k, col] = 1.0
                    W3[k*3+i, col] = 1.0
        return U3, V3, W3

    def construct_strassen_189():
        U3, V3, W3 = get_schoolbook_3x3()
        r3 = 27
        total_rank = 7 * r3
        U6 = np.zeros((36, total_rank))
        V6 = np.zeros((36, total_rank))
        W6 = np.zeros((36, total_rank))

        def add_strassen_term(start_col, a_blocks, b_blocks, c_blocks):
            # a_blocks, b_blocks, c_blocks are lists of (block_index, coefficient)
            # block_index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            
            # Pre-compute the linear combinations for A, B, C parts
            # The term computes P = M * N where M = sum_a coeff_a * A_block_a
            # and N = sum_b coeff_b * B_block_b
            # And P contributes to C as sum_c coeff_c * C_block_c
            
            # We iterate over the 27 terms of the 3x3 decomposition
            for r in range(r3):
                u_vec = U3[:, r]
                v_vec = V3[:, r]
                w_vec = W3[:, r]
                
                # u_vec corresponds to the A-part of the 3x3 multiplication
                # v_vec corresponds to the B-part
                # w_vec corresponds to the C-part
                
                # For each local (i,j) in the 3x3 A-matrix:
                for i in range(3):
                    for j in range(3):
                        local_u_idx = i * 3 + j
                        u_val = u_vec[local_u_idx]
                        if u_val == 0:
                            continue
                        
                        # Accumulate contribution to global U6
                        for bi, coeff in a_blocks:
                            row_block = bi // 2
                            col_block = bi % 2
                            global_i = row_block * 3 + i
                            global_j = col_block * 3 + j
                            global_u_idx = global_i * 6 + global_j
                            U6[global_u_idx, start_col + r] += coeff * u_val
                
                # For each local (j,k) in the 3x3 B-matrix:
                for j in range(3):
                    for k in range(3):
                        local_v_idx = j * 3 + k
                        v_val = v_vec[local_v_idx]
                        if v_val == 0:
                            continue
                        
                        # Accumulate contribution to global V6
                        for bi, coeff in b_blocks:
                            row_block = bi // 2
                            col_block = bi % 2
                            global_j = row_block * 3 + j
                            global_k = col_block * 3 + k
                            global_v_idx = global_j * 6 + global_k
                            V6[global_v_idx, start_col + r] += coeff * v_val
                
                # For each local (k,i) in the 3x3 C-matrix:
                for k in range(3):
                    for i in range(3):
                        local_w_idx = k * 3 + i
                        w_val = w_vec[local_w_idx]
                        if w_val == 0:
                            continue
                        
                        # Accumulate contribution to global W6
                        # Note: W index is k*n + i. 
                        # In the block structure, the "row" of C is determined by the output block's row index.
                        # The "col" of C is determined by the output block's col index.
                        # Local i is the row within the block, local k is the col within the block.
                        # So global row = block_row * 3 + i, global col = block_col * 3 + k.
                        # Global W index = global_col * 6 + global_row.
                        for ci, coeff in c_blocks:
                            row_block = ci // 2
                            col_block = ci % 2
                            global_k = col_block * 3 + k
                            global_i = row_block * 3 + i
                            global_w_idx = global_k * 6 + global_i
                            W6[global_w_idx, start_col + r] += coeff * w_val

        # Strassen Formulas for 2x2 block multiplication:
        # P1 = (A11 + A22) * (B11 + B22)
        # P2 = (A21 + A22) * B11
        # P3 = A11 * (B12 - B22)
        # P4 = A22 * (B21 - B11)
        # P5 = (A11 + A12) * B22
        # P6 = (A21 - A11) * (B11 + B12)
        # P7 = (A12 - A22) * (B21 + B22)
        
        # C11 = P1 - P5 + P4 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # Block indices: 0=A11, 1=A12, 2=A21, 3=A22
        # Same for B and C.

        # P1
        add_strassen_term(0, 
                          a_blocks=[(0, 1), (3, 1)], 
                          b_blocks=[(0, 1), (3, 1)], 
                          c_blocks=[(0, 1), (3, -1)]) # Wait, C11 gets +P1, C22 gets +P1.
        # Correction: The function `add_strassen_term` adds the *entire* product P to the C blocks specified.
        # So for P1, it contributes +1 to C11 (block 0) and +1 to C22 (block 3).
        # Let's re-run with correct c_blocks.
        
        U6.fill(0); V6.fill(0); W6.fill(0) # Reset

        # P1 = (A11 + A22) * (B11 + B22)
        # Contributes +1 to C11, +1 to C22
        add_strassen_term(0, 
                          a_blocks=[(0, 1), (3, 1)], 
                          b_blocks=[(0, 1), (3, 1)], 
                          c_blocks=[(0, 1), (3, 1)])

        # P2 = (A21 + A22) * B11
        # Contributes +1 to C21, -1 to C22
        add_strassen_term(r3, 
                          a_blocks=[(2, 1), (3, 1)], 
                          b_blocks=[(0, 1)], 
                          c_blocks=[(2, 1), (3, -1)])

        # P3 = A11 * (B12 - B22)
        # Contributes +1 to C12, +1 to C22
        add_strassen_term(2*r3, 
                          a_blocks=[(0, 1)], 
                          b_blocks=[(1, 1), (3, -1)], 
                          c_blocks=[(1, 1), (3, 1)])

        # P4 = A22 * (B21 - B11)
        # Contributes +1 to C11, +1 to C21
        add_strassen_term(3*r3, 
                          a_blocks=[(3, 1)], 
                          b_blocks=[(2, 1), (0, -1)], 
                          c_blocks=[(0, 1), (2, 1)])

        # P5 = (A11 + A12) * B22
        # Contributes -1 to C11, +1 to C12
        add_strassen_term(4*r3, 
                          a_blocks=[(0, 1), (1, 1)], 
                          b_blocks=[(3, 1)], 
                          c_blocks=[(0, -1), (1, 1)])

        # P6 = (A21 - A11) * (B11 + B12)
        # Contributes +1 to C22
        add_strassen_term(5*r3, 
                          a_blocks=[(2, 1), (0, -1)], 
                          b_blocks=[(0, 1), (1, 1)], 
                          c_blocks=[(3, 1)])

        # P7 = (A12 - A22) * (B21 + B22)
        # Contributes +1 to C11
        add_strassen_term(6*r3, 
                          a_blocks=[(1, 1), (3, -1)], 
                          b_blocks=[(2, 1), (3, 1)], 
                          c_blocks=[(0, 1)])

        return U6, V6, W6

    def verify_decomposition(U, V, W):
        U_reshaped = U.reshape(6, 6, -1)
        V_reshaped = V.reshape(6, 6, -1)
        W_reshaped = W.reshape(6, 6, -1)
        try:
            # Tensor decomposition check:
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # Target T[i,j,k] = 1 if i,j,k are valid indices for matrix mult?
            # The matrix multiplication tensor M[i,j,k] = 1 if the operation is A[i,j] * B[j,k] -> C[i,k].
            # Wait, the standard tensor for matrix mult M_{n} has indices (i, j, k) where:
            # A is (i, j), B is (j, k), C is (i, k).
            # The task says W[k*n+i, r]. This implies W is indexed by (k, i) where k is col of C, i is row of C.
            # So the term is U[i, j, r] * V[j, k, r] * W[k, i, r].
            # The sum over r should give 1 if the indices form a valid chain i->j->k?
            # Actually, the decomposition is:
            # C[i, k] = sum_j A[i, j] B[j, k]
            # This is a linear map. The tensor T[i, j, k] represents the coefficient of A[i, j] B[j, k] in C[i, k].
            # So T[i, j, k] = 1 for all i, j, k.
            # Let's check the einsum: 'ijr,jkr,kir->ijk'
            # U is (6, 6, R), V is (6, 6, R), W is (6, 6, R).
            # U[i, j, r], V[j, k, r], W[k, i, r].
            # Sum over r. Result is (6, 6, 6) tensor.
            # If the decomposition is correct, this tensor should be all 1s.
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            target = np.ones((6, 6, 6))
            return np.allclose(reconstructed, target, atol=1e-9)
        except:
            return False

    def get_schoolbook_6x6():
        rank = 216
        U = np.zeros((36, rank))
        V = np.zeros((36, rank))
        W = np.zeros((36, rank))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    idx = i*6 + j
                    jk = j*6 + k
                    ki = k*6 + i
                    col = i*36 + j*6 + k
                    U[idx, col] = 1.0
                    V[jk, col] = 1.0
                    W[ki, col] = 1.0
        return U, V, W

    # Check deadline
    if time.time() - start_time > DEADLINE:
        U, V, W = get_schoolbook_6x6()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U189, V189, W189 = construct_strassen_189()
    
    if time.time() - start_time > DEADLINE:
        return {"U": U189.tolist(), "V": V189.tolist(), "W": W189.tolist()}

    valid_189 = verify_decomposition(U189, V189, W189)
    
    if valid_189:
        return {"U": U189.tolist(), "V": V189.tolist(), "W": W189.tolist()}
    else:
        U, V, W = get_schoolbook_6x6()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
