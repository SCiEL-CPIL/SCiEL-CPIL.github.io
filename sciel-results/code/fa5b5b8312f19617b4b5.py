import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U, V, W are 4x7 matrices
    # Indices for U: 00, 01, 10, 11
    # Indices for V: 00, 01, 10, 11
    # Indices for W: 00, 01, 10, 11 (k*2+i)
    
    U_2 = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 0, 1, 0], # P5
        [-1, 1, 0, 0], # P6
        [0, -1, 0, 1]  # P7
    ]
    
    V_2 = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1], # P3
        [0, 0, 1, -1], # P4
        [0, 0, 0, 1], # P5
        [1, 0, 1, 0], # P6
        [0, 1, 0, 1]  # P7
    ]
    
    W_2 = [
        [1, 0, 0, 1], # P1
        [0, 1, 0, -1], # P2
        [0, 0, 1, 1], # P3
        [1, 1, 0, 0], # P4
        [-1, 0, 0, 1], # P5
        [0, 1, 0, 0], # P6
        [-1, 0, 1, 0]  # P7
    ]
    
    # Convert to numpy for easier manipulation
    U_2 = np.array(U_2, dtype=float).T # 4x7
    V_2 = np.array(V_2, dtype=float).T # 4x7
    W_2 = np.array(W_2, dtype=float).T # 4x7
    
    # Construct 4x4x4 decomposition by expanding blocks
    # Block size 2x2. 
    # U_4 will be 16x49.
    # The 16 rows of U_4 correspond to the 16 entries of the 4x4 matrix A.
    # Row index r in U_4: i*4+j.
    # The 4 blocks of A are:
    # A00: rows 0,1 cols 0,1 -> indices 0,1,4,5
    # A01: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # A10: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # A11: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    # Map block index (0..3) to the 4 row indices in the 16x? matrix
    block_rows_U = {
        0: [0, 1, 4, 5],   # A00
        1: [2, 3, 6, 7],   # A01
        2: [8, 9, 12, 13], # A10
        3: [10, 11, 14, 15]# A11
    }
    
    # For V (B matrix, 4x4), blocks are:
    # B00: rows 0,1 cols 0,1 -> indices 0,1,4,5 (j*4+k)
    # B01: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # B10: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # B11: rows 2,3 cols 2,3 -> indices 10,11,14,15
    block_rows_V = {
        0: [0, 1, 4, 5],   # B00
        1: [2, 3, 6, 7],   # B01
        2: [8, 9, 12, 13], # B10
        3: [10, 11, 14, 15]# B11
    }
    
    # For W (C matrix, 4x4), indices are k*4+i.
    # C00: k=0,1, i=0,1 -> indices 0,1,4,5
    # C01: k=0,1, i=2,3 -> indices 2,3,6,7
    # C10: k=2,3, i=0,1 -> indices 8,9,12,13
    # C11: k=2,3, i=2,3 -> indices 10,11,14,15
    block_rows_W = {
        0: [0, 1, 4, 5],   # C00
        1: [2, 3, 6, 7],   # C01
        2: [8, 9, 12, 13], # C10
        3: [10, 11, 14, 15]# C11
    }
    
    def expand_block_matrix(M_block, block_rows_map):
        # M_block is 4x7 (4 blocks, 7 terms)
        # Result is 16x49 (16 entries, 7*7 terms)
        # Wait, the 49 terms come from 7 block-terms x 7 block-terms?
        # No. The Strassen algorithm for 4x4x4 uses 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication has rank 7.
        # So for each of the 7 block-products, we have 7 rank-1 terms.
        # Total 49 terms.
        
        # M_block is the coefficient matrix for the blocks.
        # We need to expand each block coefficient into the 4 entries.
        # And we need to combine the 7 block-terms with the 7 internal terms.
        
        # Actually, the tensor decomposition is a sum of rank-1 tensors.
        # The 49 rank-1 tensors are formed by taking the tensor product of the block-level decomposition and the internal decomposition?
        # No.
        # The block-level Strassen gives 7 products $P_r = U_{block,r} \cdot V_{block,r}$.
        # Each $P_r$ is a 2x2 matrix multiplication of the resulting block?
        # No, $P_r$ is the result of multiplying two 2x2 matrices (which are linear combinations of the original blocks).
        # Wait, Strassen computes the product of two 2x2 matrices using 7 scalar multiplications.
        # Here, the "scalars" are 2x2 matrices.
        # So we need 7 matrix multiplications of 2x2 matrices.
        # Each 2x2 matrix multiplication can be decomposed into 7 rank-1 tensors.
        # So for each of the 7 Strassen steps, we perform a rank-7 decomposition of the 2x2 matrix multiplication.
        # The inputs to these 2x2 multiplications are linear combinations of the original 4x4 blocks.
        
        # Let's construct the 49 vectors directly.
        # For each Strassen step r (1..7):
        #   We have linear forms $L_A^{(r)}$ and $L_B^{(r)}$ on the 4 blocks of A and B.
        #   These are the rows of U_2 and V_2.
        #   The product is $M_r = L_A^{(r)} \cdot L_B^{(r)}$.
        #   $M_r$ is a 2x2 matrix.
        #   We need to decompose the multiplication $X \cdot Y = Z$ for 2x2 matrices, where X, Y are the inputs.
        #   But X and Y are not independent variables, they are linear combinations.
        #   However, the tensor decomposition is linear.
        #   We can just take the 7 rank-1 terms of the 2x2 multiplication and substitute the linear forms.
        
        # The 7 rank-1 terms for 2x2 multiplication are given by U_2, V_2, W_2 (4x7).
        # Let's call the internal terms $t=1..7$.
        # The total term is $(r, t)$.
        # $U_{total}[i, (r,t)] = \sum_{block} U_2[block, r] \cdot U_2[block, t]$? No.
        
        # Correct Logic:
        # The multiplication of two 2x2 matrices $X, Y$ is $Z = XY$.
        # The tensor is $\sum_{t=1}^7 u_t \otimes v_t \otimes w_t$.
        # Here $X$ and $Y$ are linear combinations of the blocks of A and B.
        # $X = \sum_{b=0}^3 U_2[b, r] A_b$.
        # $Y = \sum_{b=0}^3 V_2[b, r] B_b$.
        # The rank-1 tensor for the internal term $t$ is $u_t \otimes v_t \otimes w_t$.
        # When we substitute $X$ and $Y$, the coefficient for $A_b$ in the $u$ part is $U_2[b, r] u_t[i]$.
        # So the new $U$ vector for term $(r,t)$ is:
        # $U_{new}[\text{entry of } A_b, (r,t)] = U_2[b, r] \cdot u_t[\text{entry of } X]$.
        
        U_4 = np.zeros((16, 49))
        V_4 = np.zeros((16, 49))
        W_4 = np.zeros((16, 49))
        
        for r in range(7):
            for t in range(7):
                col_idx = r * 7 + t
                # Expand U
                for b in range(4):
                    coeff_u = U_2[b, r]
                    if coeff_u != 0:
                        rows = block_rows_U[b]
                        # u_t is the t-th column of U_2 (which is 4x7, so column t)
                        # U_2 is 4x7. u_t is U_2[:, t].
                        for i, row_idx in enumerate(rows):
                            U_4[row_idx, col_idx] = coeff_u * U_2[i, t]
                
                # Expand V
                for b in range(4):
                    coeff_v = V_2[b, r]
                    if coeff_v != 0:
                        rows = block_rows_V[b]
                        for i, row_idx in enumerate(rows):
                            V_4[row_idx, col_idx] = coeff_v * V_2[i, t]
                
                # Expand W
                # W_2 is 4x7. w_t is W_2[:, t].
                # The block index for W corresponds to the position in the 2x2 result.
                # The result of the 2x2 multiplication is a 2x2 matrix.
                # The linear form in C is given by W_2.
                # W_2[b, t] is the coefficient for the block $C_b$ in the t-th internal term.
                for b in range(4):
                    coeff_w = W_2[b, t]
                    if coeff_w != 0:
                        rows = block_rows_W[b]
                        for i, row_idx in enumerate(rows):
                            W_4[row_idx, col_idx] = coeff_w * W_2[i, t]
                            
        return U_4, V_4, W_4

    U_4, V_4, W_4 = expand_block_matrix(None, None) # Placeholder, logic inside
    
    # Now construct 4x4x8 decomposition
    # We have two 4x4x4 multiplications.
    # One for columns 0-3, one for columns 4-7.
    # U_8 will be 16x98.
    # V_8 will be 32x98.
    # W_8 will be 32x98.
    
    U_8 = np.zeros((16, 98))
    V_8 = np.zeros((32, 98))
    W_8 = np.zeros((32, 98))
    
    # First half (cols 0-3)
    U_8[:, 0:49] = U_4
    # V_4 is 16x49. It corresponds to B entries 0-15 (cols 0-3).
    # V_8 rows 0-15 correspond to B cols 0-3.
    V_8[0:16, 0:49] = V_4
    # W_4 is 16x49. It corresponds to C entries 0-15 (cols 0-3).
    # W_8 rows 0-15 correspond to C cols 0-3.
    W_8[0:16, 0:49] = W_4
    
    # Second half (cols 4-7)
    # The multiplication is the same, but B and C indices are shifted.
    # U is the same (A is the same).
    U_8[:, 49:98] = U_4
    
    # V_8 rows 16-31 correspond to B cols 4-7.
    # The pattern of V_4 applies to these rows.
    V_8[16:32, 49:98] = V_4
    
    # W_8 rows 16-31 correspond to C cols 4-7.
    # The pattern of W_4 applies to these rows.
    W_8[16:32, 49:98] = W_4
    
    # Convert to lists of lists with integer/half-integer entries
    # The entries are sums of products of -1, 0, 1. So they are integers.
    def to_list(mat):
        return [[int(x) if x == int(x) else float(x) for x in row] for row in mat]
    
    U_list = to_list(U_8)
    V_list = to_list(V_8)
    W_list = to_list(W_8)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
