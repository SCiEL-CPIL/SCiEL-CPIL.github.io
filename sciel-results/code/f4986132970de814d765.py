"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 6x6 matrix multiplication.
    Strategy: Block-Strassen with 3x3 schoolbook blocks.
    - Split 6x6 into 2x2 blocks of 3x3 matrices.
    - Use Strassen's 7 multiplications for the 2x2 block structure.
    - Use schoolbook (rank 27) for each 3x3 multiplication.
    - Total rank = 7 * 27 = 189.
    """
    n, m, p = DIMENSIONS
    if (n, m, p) != (6, 6, 6):
        # Fallback for other sizes (not expected in this task)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Block size
    b = 3
    
    # 1. Define Schoolbook decomposition for b x b matrices
    # For 3x3 matrices, rank is 27.
    # U_S: (b*b, b*b*b) -> (9, 27)
    # V_S: (b*b, b*b*b) -> (9, 27)
    # W_S: (b*b, b*b*b) -> (9, 27)
    # Indexing:
    # U[i*b+j, col] = 1 if col corresponds to A[i][j]
    # V[j*b+k, col] = 1 if col corresponds to B[j][k]
    # W[k*b+i, col] = 1 if col corresponds to C[i][k]
    # Let col = i*b*b + j*b + k
    rank_s = b * b * b
    U_S = np.zeros((b * b, rank_s))
    V_S = np.zeros((b * b, rank_s))
    W_S = np.zeros((b * b, rank_s))
    
    for i in range(b):
        for j in range(b):
            for k in range(b):
                col = i * b * b + j * b + k
                U_S[i * b + j, col] = 1.0
                V_S[j * b + k, col] = 1.0
                W_S[k * b + i, col] = 1.0

    # 2. Define Strassen decomposition for 2x2 block matrices
    # Blocks: 00, 01, 10, 11
    # Strassen steps:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # We represent the 2x2 block multiplication as a tensor decomposition.
    # Each "step" s involves a linear combination of A blocks and a linear combination of B blocks.
    # Since the base decomposition (schoolbook) is linear in A and B, we can just apply the same
    # linear combinations to the rows of U_S and columns of V_S.
    # And the W_S matrix is reused for each step, but combined with the C-block coefficients.
    
    # Let's define the coefficients for each step.
    # A_blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B_blocks: 0=B00, 1=B01, 2=B10, 3=B11
    # C_blocks: 0=C00, 1=C01, 2=C10, 3=C11
    
    # For each step s, we have:
    # A_s = sum_alpha a_alpha * A_alpha
    # B_s = sum_beta b_beta * B_beta
    # C_s = sum_gamma c_gamma * C_gamma (this is how the result M_s contributes to C)
    
    # However, in the tensor decomposition framework:
    # U_total[i, (s, t)] = U_S[i_block, t] * coeff_A[i_block, s]
    # V_total[j, (s, t)] = V_S[j_block, t] * coeff_B[j_block, s]
    # W_total[k, (s, t)] = W_S[k_block, t] * coeff_C[k_block, s]
    
    # Wait, the W part is slightly different.
    # In the standard tensor decomposition C = sum_r U_r * V_r * W_r (outer products),
    # the W part indicates which output element is being computed.
    # In Strassen, M_s is an intermediate matrix. It contributes to multiple C blocks.
    # C_block_gamma += c_gamma * M_s
    # So for a specific term t in the schoolbook decomposition of M_s,
    # it contributes to C_block_gamma with weight c_gamma.
    # Thus, W_total[k_global, (s,t)] = W_S[k_block, t] * c_gamma
    
    # Let's define the steps:
    # s=0: M1. A: {0:1, 3:1}, B: {0:1, 3:1}. C: {0:1, 1:1, 2:1, 3:1} -- NO.
    # M1 contributes to C00(+1), C11(+1). Wait.
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # So:
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(-1), C10(+1)
    # M5: C01(+1)
    # M6: C00(+1)
    # M7: C00(-1), C11(+1)
    
    # Let's verify M5: (A00+A01)B11.
    # C01 = M3 + M5. Yes.
    # C10 = M2 + M4. Yes.
    
    # Definitions:
    # A_coeffs[s][block_idx]
    # B_coeffs[s][block_idx]
    # C_coeffs[s][block_idx]
    
    A_coeffs = [
        {0: 1, 3: 1},      # M1: A00+A11
        {2: 1, 3: 1},      # M2: A10+A11
        {0: 1},            # M3: A00
        {3: 1},            # M4: A11
        {0: 1, 1: 1},      # M5: A00+A01
        {2: 1, 0: -1},     # M6: A10-A00
        {1: 1, 3: -1},     # M7: A01-A11
    ]
    
    B_coeffs = [
        {0: 1, 3: 1},      # M1: B00+B11
        {0: 1},            # M2: B00
        {1: 1, 3: -1},     # M3: B01-B11
        {2: 1, 0: -1},     # M4: B10-B00
        {3: 1},            # M5: B11
        {0: 1, 1: 1},      # M6: B00+B01
        {2: 1, 3: 1},      # M7: B10+B11
    ]
    
    C_coeffs = [
        {0: 1, 3: 1},      # M1: C00+1, C11+1
        {2: 1, 3: -1},     # M2: C10+1, C11-1
        {1: 1, 3: 1},      # M3: C01+1, C11+1
        {0: -1, 2: 1},     # M4: C00-1, C10+1
        {1: 1},            # M5: C01+1
        {0: 1},            # M6: C00+1
        {0: -1, 3: 1},     # M7: C00-1, C11+1
    ]
    
    # Now construct U, V, W
    # Total rank R = 7 * 27 = 189
    R = 7 * rank_s
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Mapping from block index to global indices
    # Block 0 (00): rows 0-2, cols 0-2
    # Block 1 (01): rows 0-2, cols 3-5
    # Block 2 (10): rows 3-5, cols 0-2
    # Block 3 (11): rows 3-5, cols 3-5
    
    # U is indexed by (row, col) of A. A is 6x6.
    # A[i][j].
    # If i < 3, row block is 0. Else 1.
    # If j < 3, col block is 0. Else 1.
    # Block index = 2*row_block + col_block?
    # 00: 0, 01: 1, 10: 2, 11: 3.
    # Yes.
    
    # V is indexed by (row, col) of B. B is 6x6.
    # B[i][j].
    # Same block mapping.
    
    # W is indexed by (row, col) of C?
    # The task says W[k*n+i, r] where C[i][k] is the output.
    # So W index is (k, i).
    # C[i][k].
    # If i < 3, row block is 0. Else 1.
    # If k < 3, col block is 0. Else 1.
    # Block index = 2*row_block + col_block.
    
    for s in range(7):
        for t in range(rank_s):
            col_idx = s * rank_s + t
            
            # Get the schoolbook coefficients for this term t
            # U_S[i_s, t] is non-zero for exactly one i_s (which is i*b+j)
            # V_S[j_s, t] is non-zero for exactly one j_s (which is j*b+k)
            # W_S[k_s, t] is non-zero for exactly one k_s (which is k*b+i)
            
            # Find the non-zero entries in U_S[:, t], V_S[:, t], W_S[:, t]
            # Since they are 0/1 and sparse, we can just loop or use argmax if unique.
            # But it's safer to iterate.
            
            # Actually, we can precompute the indices for each t.
            # i_s = t // (b*b)
            # j_s = (t // b) % b
            # k_s = t % b
            
            i_s = t // (b * b)
            j_s = (t // b) % b
            k_s = t % b
            
            # i_s is the local row index in the 3x3 block (0-2)
            # j_s is the local col index in A / row index in B (0-2)
            # k_s is the local col index in B (0-2)
            
            # Wait, U_S[i*b+j, col] = 1.
            # So the global index in the block is i*b+j.
            # i_s is i, j_s is j.
            
            # For each A block that participates in step s:
            for a_block, a_coeff in A_coeffs[s].items():
                if a_coeff == 0: continue
                # Determine global row and col ranges for this A block
                # a_block: 0=00, 1=01, 2=10, 3=11
                row_block = a_block // 2
                col_block = a_block % 2
                
                # Global row range: [row_block*b, (row_block+1)*b)
                # Global col range: [col_block*b, (col_block+1)*b)
                
                # The local row i_s and local col j_s correspond to global:
                global_i = row_block * b + i_s
                global_j = col_block * b + j_s
                
                # U index is i*m + j
                U[global_i * m + global_j, col_idx] = a_coeff
            
            # For each B block that participates in step s:
            for b_block, b_coeff in B_coeffs[s].items():
                if b_coeff == 0: continue
                row_block = b_block // 2
                col_block = b_block % 2
                
                # V is indexed by (row, col) of B.
                # Local row is j_s (since V_S[j*b+k, t] is non-zero, j is the row of B)
                # Local col is k_s
                global_i = row_block * b + j_s
                global_j = col_block * b + k_s
                
                # V index is i*p + j
                V[global_i * p + global_j, col_idx] = b_coeff
            
            # For each C block that participates in step s:
            for c_block, c_coeff in C_coeffs[s].items():
                if c_coeff == 0: continue
                row_block = c_block // 2
                col_block = c_block % 2
                
                # W is indexed by (k, i) where C[i][k].
                # Local row i (of C) is i_s
                # Local col k (of C) is k_s
                global_i = row_block * b + i_s
                global_k = col_block * b + k_s
                
                # W index is k*n + i
                W[global_k * n + global_i, col_idx] = c_coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
