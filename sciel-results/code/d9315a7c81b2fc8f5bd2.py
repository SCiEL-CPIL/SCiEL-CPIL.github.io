import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Try Strassen for 2x2 blocks of 3x3 matrices
    # This gives rank 7 * 21 = 147 if we use BCLR for 3x3
    
    # First, let's define a valid rank-21 decomposition for 3x3 matrix multiplication
    # BCLR (Bini-Capovani-Lotti-Romani) decomposition
    
    # The BCLR coefficients for 3x3 matrix multiplication
    # U, V, W are 9x21 matrices where:
    # U[i*3+j, c] = u_{ijc}
    # V[j*3+k, c] = v_{jkc}  
    # W[k*3+i, c] = w_{kic}
    
    # I'll use a known valid rank-21 decomposition
    # This is the BCLR decomposition with integer coefficients
    
    BCLR_U = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
    
    # This is just the identity/schoolbook approach for 3x3 with rank 27.
    # I need to actually implement a rank-21 decomposition.
    
    # Let me use the known BCLR decomposition properly.
    # For 3x3 matrix multiplication, there's a rank-21 decomposition.
    
    # Actually, let me just use the schoolbook for now and try to optimize.
    # The schoolbook for 6x6 has rank 216.
    # If I can get rank 147 using Strassen + BCLR, that would be an improvement.
    
    # Let me implement Strassen for 2x2 blocks of 3x3 matrices.
    # Each 2x2 Strassen term requires multiplying two 3x3 matrices.
    # Using schoolbook for 3x3: rank = 7 * 27 = 189
    # Using BCLR for 3x3: rank = 7 * 21 = 147
    
    # I need the BCLR coefficients. Let me use a known valid rank-21 decomposition.
    
    # Here's a valid rank-21 decomposition for 3x3 matrix multiplication:
    # This uses the approach from the BCLR paper.
    
    # For now, let me use the block Strassen approach with schoolbook 3x3
    # to get rank 189, which is better than 216.
    
    if time.time() > deadline:
        # Return schoolbook as fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Block Strassen: partition 6x6 into four 3x3 blocks
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = A*B where C_ij are 3x3 matrices
    
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is a product of two 3x3 matrices (linear combinations).
    # We'll use schoolbook for each 3x3 multiplication.
    
    # Define the Strassen structure
    # For each of the 7 M terms, we have a linear combination of A-blocks and B-blocks
    
    # Let's define the coefficients:
    # A-blocks: A11 (rows 0-2, cols 0-2), A12 (rows 0-2, cols 3-5), 
    #           A21 (rows 3-5, cols 0-2), A22 (rows 3-5, cols 3-5)
    # B-blocks: B11 (rows 0-2, cols 0-2), B12 (rows 0-2, cols 3-5),
    #           B21 (rows 3-5, cols 0-2), B22 (rows 3-5, cols 3-5)
    
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # For each M term, we need to express the linear combinations.
    
    # Let's build the U, V, W matrices.
    # Total rank = 7 * 27 = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Block indices:
    # A11: i in [0,2], j in [0,2]
    # A12: i in [0,2], j in [3,5]
    # A21: i in [3,5], j in [0,2]
    # A22: i in [3,5], j in [3,5]
    
    # For U: U[i*6+j, c] gives coefficient of A[i,j] in the c-th term
    # For V: V[j*6+k, c] gives coefficient of B[j,k] in the c-th term
    # For W: W[k*6+i, c] gives coefficient of C[k,i] in the c-th term (trace convention: W[k*n+i, r])
    
    # Strassen terms:
    # term 0: M1 = (A11+A22)(B11+B22)
    # term 1: M2 = (A21+A22)B11
    # term 2: M3 = A11(B12-B22)
    # term 3: M4 = A22(B21-B11)
    # term 4: M5 = (A11+A12)B22
    # term 5: M6 = (A21-A11)(B11+B12)
    # term 6: M7 = (A12-A22)(B21+B22)
    
    # For each Strassen term, we have a 3x3 matrix multiplication using schoolbook (rank 27)
    
    # Let's define the A-linear-combination coefficients for each term:
    # a_coeffs[term][block] where block is 0=A11, 1=A12, 2=A21, 3=A22
    
    a_coeffs = [
        [1, 0, 0, 1],  # M1: A11 + A22
        [0, 0, 1, 1],  # M2: A21 + A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11 + A12
        [-1, 0, 1, 0], # M6: A21 - A11
        [0, 1, 0, -1], # M7: A12 - A22
    ]
    
    # b_coeffs[term][block] where block is 0=B11, 1=B12, 2=B21, 3=B22
    b_coeffs = [
        [1, 0, 0, 1],  # M1: B11 + B22
        [1, 0, 0, 0],  # M2: B11
        [0, 1, 0, -1], # M3: B12 - B22
        [0, 0, 1, -1], # M4: B21 - B11
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11 + B12
        [0, 0, 1, 1],  # M7: B21 + B22
    ]
    
    # For each term t (0-6) and each schoolbook sub-term s (0-26):
    # s corresponds to (i_sub, j_sub, k_sub) where i_sub, j_sub, k_sub in [0,2]
    # The A-element is at global position (row_block_start + i_sub, col_block_start + j_sub)
    # based on which A-block is used.
    
    # Actually, let me think more carefully.
    
    # For term t, the A-linear-combination is: sum over blocks b of a_coeffs[t][b] * A_b
    # where A_b is a 3x3 block.
    # The element A_b[i_sub, j_sub] corresponds to global A[i_global, j_global]
    # where i_global and j_global depend on the block.
    
    # Block A11: i_global = i_sub, j_global = j_sub
    # Block A12: i_global = i_sub, j_global = 3 + j_sub
    # Block A21: i_global = 3 + i_sub, j_global = j_sub
    # Block A22: i_global = 3 + i_sub, j_global = 3 + j_sub
    
    # Similarly for B-blocks:
    # Block B11: i_global = i_sub, j_global = j_sub
    # Block B12: i_global = i_sub, j_global = 3 + j_sub
    # Block B21: i_global = 3 + i_sub, j_global = j_sub
    # Block B22: i_global = 3 + i_sub, j_global = 3 + j_sub
    
    # For the W matrix (output C):
    # C = A*B, so C[i_global, j_global] = sum over k_global of A[i_global, k_global] * B[k_global, j_global]
    
    # But with Strassen, we have:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for C11[i_sub, j_sub]:
    # C11[i_sub, j_sub] = M1[i_sub, j_sub] + M4[i_sub, j_sub] - M5[i_sub, j_sub] + M7[i_sub, j_sub]
    
    # Each M_t[i_sub, j_sub] = sum over k_sub of (A_comb_t[i_sub, k_sub]) * (B_comb_t[k_sub, j_sub])
    
    # Where A_comb_t is the linear combination of A-blocks for term t,
    # and B_comb_t is the linear combination of B-blocks for term t.
    
    # So for a specific (i_sub, j_sub) in C11:
    # C11[i_sub, j_sub] = sum over k_sub of [sum over a_blocks a_coeffs[t][a_block] * A_a_block[i_sub, k_sub]] * [sum over b_blocks b_coeffs[t][b_block] * B_b_block[k_sub, j_sub]]
    
    # This expands to a sum over a_blocks, b_blocks, k_sub of
    # a_coeffs[t][a_block] * b_coeffs[t][b_block] * A_a_block[i_sub, k_sub] * B_b_block[k_sub, j_sub]
    
    # For the tensor decomposition, each term is a rank-1 tensor:
    # U[global_A_idx, col] * V[global_B_idx, col] * W[global_C_idx, col]
    
    # So for each term t, a_block, b_block, k_sub, we have one column.
    # But that's 7 * 4 * 4 * 3 = 336 columns, which is too many.
    
    # I need to be more careful. The schoolbook approach for each M_t should use rank 27.
    
    # Let me reconsider. For each Strassen term t, I need to compute M_t = A_comb_t * B_comb_t
    # where A_comb_t and B_comb_t are 3x3 matrices.
    
    # Using schoolbook for this 3x3 multiplication:
    # M_t[i_sub, j_sub] = sum_{k_sub=0}^{2} A_comb_t[i_sub, k_sub] * B_comb_t[k_sub, j_sub]
    
    # A_comb_t[i_sub, k_sub] = sum over a_blocks of a_coeffs[t][a_block] * A_a_block[i_sub, k_sub]
    # B_comb_t[k_sub, j_sub] = sum over b_blocks of b_coeffs[t][b_block] * B_b_block[k_sub, j_sub]
    
    # So M_t[i_sub, j_sub] = sum_{k_sub} sum_{a_blocks} sum_{b_blocks} 
    #   a_coeffs[t][a_block] * b_coeffs[t][b_block] * A_a_block[i_sub, k_sub] * B_b_block[k_sub, j_sub]
    
    # For the tensor decomposition, each (t, k_sub, a_block, b_block) gives a term.
    # But we can factor this differently.
    
    # Actually, the standard approach is:
    # For each Strassen term t, use a rank-r decomposition for the 3x3 multiplication.
    # If we use schoolbook (rank 27), then for each t, we have 27 columns.
    # Total rank = 7 * 27 = 189.
    
    # For each t and each schoolbook sub-term (i_sub, j_sub, k_sub):
    # The A-element is A_a_block[i_sub, k_sub] for some a_block, but we need to sum over a_blocks.
    
    # Hmm, this is getting complex. Let me just directly construct the matrices.
    
    # For column c = t * 27 + s, where t in [0,6] and s in [0,26]:
    # s = i_sub * 9 + j_sub * 3 + k_sub (or some other encoding)
    
    # Actually, let me use s = i_sub * 3 * 3 + k_sub * 3 + j_sub or similar.
    
    # Let me use: for schoolbook 3x3, the (i_sub, k_sub) pair selects the A-element,
    # and (k_sub, j_sub) selects the B-element, contributing to C[i_sub, j_sub].
    
    # So for sub-term s, let's say i_sub = s // 9, k_sub = (s // 3) % 3, j_sub = s % 3.
    # Wait, that's 3*3*3 = 27 combinations, which is correct.
    
    # Actually, for schoolbook: C[i,j] = sum_k A[i,k] * B[k,j]
    # So each term is (i, k, j) -> A[i,k] * B[k,j]
    # There are 3*3*3 = 27 such terms.
    
    # Let me use: s encodes (i_sub, k_sub, j_sub) as s = i_sub * 9 + k_sub * 3 + j_sub
    
    for t in range(7):
        for s in range(27):
            i_sub = s // 9
            k_sub = (s // 3) % 3
            j_sub = s % 3
            
            col = t * 27 + s
            
            # A-element: A_comb_t[i_sub, k_sub] = sum over a_blocks of a_coeffs[t][a_block] * A_a_block[i_sub, k_sub]
            # So U[global_A_idx, col] = a_coeffs[t][a_block] for the appropriate global_A_idx
            
            for a_block in range(4):
                coeff_a = a_coeffs[t][a_block]
                if coeff_a == 0:
                    continue
                
                # Determine global row and col for A_a_block[i_sub, k_sub]
                if a_block == 0:  # A11
                    i_global = i_sub
                    j_global = k_sub
                elif a_block == 1:  # A12
                    i_global = i_sub
                    j_global = 3 + k_sub
                elif a_block == 2:  # A21
                    i_global = 3 + i_sub
                    j_global = k_sub
                else:  # a_block == 3, A22
                    i_global = 3 + i_sub
                    j_global = 3 + k_sub
                
                U[i_global * m + j_global][col] = coeff_a
            
            # B-element: B_comb_t[k_sub, j_sub] = sum over b_blocks of b_coeffs[t][b_block] * B_b_block[k_sub, j_sub]
            for b_block in range(4):
                coeff_b = b_coeffs[t][b_block]
                if coeff_b == 0:
                    continue
                
                # Determine global row and col for B_b_block[k_sub, j_sub]
                if b_block == 0:  # B11
                    i_global = k_sub
                    j_global = j_sub
                elif b_block == 1:  # B12
                    i_global = k_sub
                    j_global = 3 + j_sub
                elif b_block == 2:  # B21
                    i_global = 3 + k_sub
                    j_global = j_sub
                else:  # b_block == 3, B22
                    i_global = 3 + k_sub
                    j_global = 3 + j_sub
                
                V[i_global * p + j_global][col] = coeff_b
            
            # C-element: C[i_sub, j_sub] in the appropriate C-block
            # C11[i_sub, j_sub]: global C[i_sub, j_sub]
            # C12[i_sub, j_sub]: global C[i_sub, 3+j_sub]
            # C21[i_sub, j_sub]: global C[3+i_sub, j_sub]
            # C22[i_sub, j_sub]: global C[3+i_sub, 3+j_sub]
            
            # W uses trace convention: W[k*n+i, r]
            # So W[global_k * n + global_i, col]
            
            # The contribution to C depends on which C-block this Strassen term contributes to.
            # From Strassen:
            # C11 = M1 + M4 - M5 + M7  -> terms 0, 3, 4, 6 with coeffs 1, 1, -1, 1
            # C12 = M3 + M5           -> terms 2, 4 with coeffs 1, 1
            # C21 = M2 + M4           -> terms 1, 3 with coeffs 1, 1
            # C22 = M1 - M2 + M3 + M6 -> terms 0, 1, 2, 5 with coeffs 1, -1, 1, 1
            
            # So for term t, it contributes to certain C-blocks with certain coefficients.
            
            c_block_coeffs = {
                0: {0: 1, 1: 0, 2: 0, 3: 1},  # M1 -> C11(1), C22(1)
                1: {0: 0, 1: 0, 2: 1, 3: -1},  # M2 -> C21(1), C22(-1)
                2: {0: 0, 1: 1, 2: 0, 3: 1},   # M3 -> C12(1), C22(1)
                3: {0: 1, 1: 0, 2: 1, 3: 0},   # M4 -> C11(1), C21(1)
                4: {0: -1, 1: 1, 2: 0, 3: 0},  # M5 -> C11(-1), C12(1)
                5: {0: 0, 1: 0, 2: 0, 3: 1},   # M6 -> C22(1)
                6: {0: 1, 1: 0, 2: 0, 3: 0},   # M7 -> C11(1)
            }
            
            for c_block in range(4):
                coeff_c = c_block_coeffs[t][c_block]
                if coeff_c == 0:
                    continue
                
                # Determine global row and col for C_c_block[i_sub, j_sub]
                if c_block == 0:  # C11
                    i_global = i_sub
                    j_global = j_sub
                elif c_block == 1:  # C12
                    i_global = i_sub
                    j_global = 3 + j_sub
                elif c_block == 2:  # C21
                    i_global = 3 + i_sub
                    j_global = j_sub
                else:  # c_block == 3, C22
                    i_global = 3 + i_sub
                    j_global = 3 + j_sub
                
                # W[k*n + i, col] where i is the row of C, k is the col of C
                # Wait, the trace convention is W[k*n+i, r]
                # So the index is k*n + i, where i is the row index and k is the col index of C
                # So for C[i_global, j_global], the W index is j_global * n + i_global
                
                W[j_global * n + i_global][col] = coeff_c
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
