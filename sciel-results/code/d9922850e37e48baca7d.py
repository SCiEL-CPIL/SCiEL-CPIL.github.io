import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 65: Strassen for 4x4x4 (49) + Schoolbook for 4x4x1 (16)
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # U_idx: i*m + j
    # V_idx: j*p + k
    # W_idx: k*n + i
    def add_term(i, j, k, coeff, col):
        U[i * m + j][col] += coeff
        V[j * p + k][col] += coeff
        W[k * n + i][col] += coeff

    # Part 1: Strassen for 4x4x4 (k=1..4)
    # We use the recursive Strassen structure.
    # 7 blocks, each block is a 2x2 Strassen (7 scalar products).
    # Total 49 products.
    
    # Strassen coefficients for 2x2:
    # M1: (11+22)(11+22)
    # M2: (21+22)(11)
    # M3: (11)(21-22)
    # M4: (22)(11-12)
    # M5: (11+12)(22)
    # M6: (21-11)(11+12)
    # M7: (12-22)(21+22)
    
    # We need to map the 4x4 indices to 2x2 blocks.
    # Block 1: A[0:2, 0:2], B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 2:4], B[2:4, 0:2] -> C[0:2, 2:4]
    # Block 3: A[2:4, 0:2], B[0:2, 2:4] -> C[2:4, 0:2]
    # Block 4: A[2:4, 2:4], B[2:4, 2:4] -> C[2:4, 2:4]
    
    # Strassen for 4x4 uses 7 intermediate 2x2 products S1..S7
    # S1 = (A11+A22)(B11+B22)
    # S2 = (A21+A22)(B11)
    # S3 = (A11)(B21-B22)
    # S4 = (A22)(B11-B12)
    # S5 = (A11+A12)(B22)
    # S6 = (A21-A11)(B11+B12)
    # S7 = (A12-A22)(B21+B22)
    
    # Each Sk is a 2x2 matrix product.
    # We decompose each Sk into 7 scalar products using Strassen again.
    
    # Indices in 4x4:
    # A11: (0,0)..(1,1), A12: (0,2)..(1,3), A21: (2,0)..(3,1), A22: (2,2)..(3,3)
    # B11: (0,0)..(1,1), B12: (0,2)..(1,3), B21: (2,0)..(3,1), B22: (2,2)..(3,3)
    
    # Define the 7 Strassen steps for the 4x4 level
    # Each step produces a 2x2 matrix Sk.
    # The factors for Sk are linear combinations of A blocks and B blocks.
    
    # We will generate the 49 terms.
    # For each of the 7 outer Strassen steps (s=0..6):
    #   Identify the A-block combination and B-block combination.
    #   For each of the 7 inner Strassen steps (t=0..6):
    #       Identify the scalar A-indices and B-indices.
    #       Add the term to U, V, W.
    
    # Outer Strassen definitions (Block indices 1,2,3,4 correspond to 0,1,2,3 in 2x2 grid)
    # S1: A(1,1)+A(2,2) with B(1,1)+B(2,2)
    # S2: A(2,1)+A(2,2) with B(1,1)
    # S3: A(1,1) with B(2,1)-B(2,2)
    # S4: A(2,2) with B(1,1)-B(1,2)
    # S5: A(1,1)+A(1,2) with B(2,2)
    # S6: A(2,1)-A(1,1) with B(1,1)+B(1,2)
    # S7: A(1,2)-A(2,2) with B(2,1)+B(2,2)
    
    # Inner Strassen definitions (Scalar indices within 2x2 block)
    # M1: (11+22)(11+22)
    # M2: (21+22)(11)
    # M3: (11)(21-22)
    # M4: (22)(11-12)
    # M5: (11+12)(22)
    # M6: (21-11)(11+12)
    # M7: (12-22)(21+22)
    
    # Mapping block index (1..4) to row/col offset in 4x4
    # Block 1: r=0, c=0
    # Block 2: r=0, c=2
    # Block 3: r=2, c=0
    # Block 4: r=2, c=2
    
    def get_block_indices(block_id, local_r, local_c):
        # block_id 1..4
        # local_r, local_c 0..1
        if block_id == 1:
            return local_r, local_c
        elif block_id == 2:
            return local_r, local_c + 2
        elif block_id == 3:
            return local_r + 2, local_c
        elif block_id == 4:
            return local_r + 2, local_c + 2

    # Outer steps: (A_blocks, B_blocks)
    # A_blocks is a list of (block_id, coeff)
    # B_blocks is a list of (block_id, coeff)
    
    outer_steps = [
        # S1
        ([(1, 1), (4, 1)], [(1, 1), (4, 1)]),
        # S2
        ([(3, 1), (4, 1)], [(1, 1)]),
        # S3
        ([(1, 1)], [(3, 1), (4, -1)]),
        # S4
        ([(4, 1)], [(1, 1), (2, -1)]),
        # S5
        ([(1, 1), (2, 1)], [(4, 1)]),
        # S6
        ([(3, 1), (1, -1)], [(1, 1), (2, 1)]),
        # S7
        ([(2, 1), (4, -1)], [(3, 1), (4, 1)]),
    ]
    
    # Inner steps: (A_local_indices, B_local_indices)
    # A_local_indices is a list of (local_r, local_c, coeff)
    # B_local_indices is a list of (local_r, local_c, coeff)
    
    inner_steps = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # M3
        ([(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M4
        ([(1,1,1)], [(0,0,1), (0,1,-1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    col = 0
    for s_idx, (a_blocks, b_blocks) in enumerate(outer_steps):
        for t_idx, (a_locals, b_locals) in enumerate(inner_steps):
            # This term contributes to Sk.
            # We need to determine which C entry this Sk entry contributes to.
            # And with what coefficient.
            
            # First, determine the local indices of the scalar product.
            # The scalar product is sum(a_locals) * sum(b_locals)
            # But wait, the Strassen formula is linear in the blocks.
            # The term is (Sum of A parts) * (Sum of B parts).
            # This expands to a sum of scalar products.
            # However, in the tensor decomposition, each "product" in the algorithm is a single scalar multiplication.
            # So we treat (Sum A) * (Sum B) as one multiplication?
            # No, the rank is the number of scalar multiplications.
            # So each (Sum A) * (Sum B) is ONE multiplication.
            # The U and V factors will have multiple non-zero entries.
            
            # So for this term (col), we set U and V to the coefficients of the sums.
            
            # Determine the global indices for U and V.
            # U: i*m + j
            # V: j*p + k
            
            # We need to know which C entry (k, i) this product contributes to.
            # The product is part of Sk.
            # Sk is a 2x2 matrix.
            # The local indices in inner_steps determine which entry of Sk is computed?
            # No, inner_steps compute the 7 products for a 2x2 multiplication.
            # The 7 products are combined to form the 4 entries of the result matrix.
            # So we need to know which entry of Sk (local_r, local_c) is being formed by this inner step t_idx.
            
            # Strassen reconstruction for 2x2:
            # C11 = M1 + M4 - M7 - M5
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So, for t_idx 0 (M1), it contributes to C11 (+1), C22 (+1).
            # For t_idx 1 (M2), it contributes to C21 (+1), C22 (-1).
            # etc.
            
            recon = [
                [(0,0, 1), (1,1, 1)], # M1
                [(1,0, 1), (1,1,-1)], # M2
                [(0,1, 1)],           # M3
                [(0,0, 1), (1,0, 1)], # M4
                [(0,0,-1), (0,1, 1)], # M5
                [(1,1, 1)],           # M6
                [(0,0,-1)],           # M7
            ]
            
            # Now, we need to map Sk entry (local_r, local_c) to global C entry.
            # Sk is the result of the outer step s_idx.
            # The outer steps S1..S7 are combined to form C blocks.
            
            # C11 = S1 + S4 - S7 - S5
            # C12 = S3 + S5
            # C21 = S2 + S4
            # C22 = S1 - S2 + S3 + S6
            
            # s_idx 0 is S1, 1 is S2, etc.
            
            outer_recon = {
                0: [(0,0, 1), (0,1, 1), (1,0, 1), (1,1, 1)], # S1 -> C11(+), C12(+), C21(0), C22(+) ? 
                # Wait, C11 is block (0,0), C12 is (0,1), C21 is (1,0), C22 is (1,1) in 2x2 block grid.
                # S1 contributes to C11 (+1) and C22 (+1).
            }
            
            # Let's define the contribution of Sk to the 4 C-blocks.
            # Sk is a 2x2 matrix.
            # C_block(0,0) = S1(0,0) + S4(0,0) - S7(0,0) - S5(0,0) ... No.
            # The Strassen formula is:
            # C11 = S1 + S4 - S7 - S5
            # C12 = S3 + S5
            # C21 = S2 + S4
            # C22 = S1 - S2 + S3 + S6
            # Here S1..S7 are 2x2 matrices.
            # So each entry of Sk contributes to the corresponding entry of the C-blocks.
            
            # So, for a given s_idx and local entry (lr, lc) of Sk:
            # It contributes to:
            # C-block (0,0) entry (lr, lc) with coeff from outer_recon[s_idx][(0,0)]
            # C-block (0,1) entry (lr, lc) with coeff from outer_recon[s_idx][(0,1)]
            # etc.
            
            outer_contrib = [
                # S1
                {(0,0): 1, (1,1): 1},
                # S2
                {(1,0): 1, (1,1): -1},
                # S3
                {(0,1): 1, (1,1): 1},
                # S4
                {(0,0): 1, (1,0): 1},
                # S5
                {(0,0): -1, (0,1): 1},
                # S6
                {(1,1): 1},
                # S7
                {(0,0): -1},
            ]
            
            # Now, for this specific term (s_idx, t_idx):
            # It computes a specific entry of Sk?
            # No, the inner Strassen products M1..M7 are scalars.
            # They are combined to form the entries of Sk.
            # So for a given t_idx, we know which entries of Sk it contributes to (from `recon`).
            # For each such entry (lr, lc) of Sk, we know which C-blocks it contributes to (from `outer_contrib[s_idx]`).
            # And for each C-block (br, bc), the global indices are:
            # i = br*2 + lr
            # k = bc*2 + lc
            
            for (lr, lc, coeff_inner) in recon[t_idx]:
                # This term contributes to Sk[lr, lc] with coeff_inner.
                # Now find which C-blocks Sk[lr, lc] contributes to.
                for (br, bc), coeff_outer in outer_contrib[s_idx].items():
                    i = br * 2 + lr
                    k = bc * 2 + lc
                    total_coeff = coeff_inner * coeff_outer
                    
                    # Now set U, V, W for this column.
                    # U and V are determined by the linear combinations in a_blocks/a_locals and b_blocks/b_locals.
                    
                    # U: i*m + j
                    # We need to sum the contributions from all a_blocks and a_locals.
                    # But wait, the term is a single scalar product.
                    # The "A-part" of this scalar product is Sum(a_blocks * a_locals).
                    # So U[i*m + j] should be the coefficient of A[i,j] in this sum.
                    
                    u_val = 0
                    v_val = 0
                    
                    # This logic is flawed. U and V are fixed for the column.
                    # The column represents ONE multiplication: (Sum A) * (Sum B).
                    # So U and V must be set to the coefficients of the sums.
                    # And W must be set to the coefficients of the output.
                    
                    # So we should set U and V once per column, and W for all affected (k,i).
                    
                    pass # We will restructure the loop.
    
    # Restructure:
    col = 0
    for s_idx, (a_blocks, b_blocks) in enumerate(outer_steps):
        for t_idx, (a_locals, b_locals) in enumerate(inner_steps):
            # Build U and V for this column
            # U is 16x1, V is 20x1.
            # U[i,j] is coeff of A[i,j]
            # V[j,k] is coeff of B[j,k]
            
            u_vec = [0] * 16
            v_vec = [0] * 20
            
            for (block_id, coeff_b) in a_blocks:
                for (lr, lc, coeff_l) in a_locals:
                    i, j = get_block_indices(block_id, lr, lc)
                    u_vec[i * m + j] += coeff_b * coeff_l
            
            for (block_id, coeff_b) in b_blocks:
                for (lr, lc, coeff_l) in b_locals:
                    j, k = get_block_indices(block_id, lr, lc)
                    # Note: B block indices are (row, col) = (j, k)
                    v_vec[j * p + k] += coeff_b * coeff_l
            
            # Now build W for this column
            # W[k*n + i]
            w_vec = [0] * 20
            
            # Determine which (k,i) are affected and their coefficients.
            # The product is (Sum A) * (Sum B).
            # This product contributes to Sk entries via recon[t_idx].
            # Sk entries contribute to C entries via outer_contrib[s_idx].
            
            for (lr, lc, coeff_inner) in recon[t_idx]:
                for (br, bc), coeff_outer in outer_contrib[s_idx].items():
                    i = br * 2 + lr
                    k = bc * 2 + lc
                    w_vec[k * n + i] += coeff_inner * coeff_outer
            
            # Add to global U, V, W
            for idx in range(16):
                U[idx][col] = u_vec[idx]
            for idx in range(20):
                V[idx][col] = v_vec[idx]
            for idx in range(20):
                W[idx][col] = w_vec[idx]
            
            col += 1
    
    # Part 2: Schoolbook for 4x4x1 (k=5)
    # C[i,5] = sum_j A[i,j] B[j,5]
    # 16 terms.
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1 # k=4 (0-indexed)
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
