"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # We will use Strassen's algorithm to achieve rank 98.
    # 4x4 by 4x8 is two 4x4 by 4x4 multiplications.
    # Each 4x4 by 4x4 can be done with rank 49 using Strassen.
    
    # The decomposition for 4x4 by 4x4:
    # Outer Strassen: 7 products of 2x2 blocks.
    # Inner Strassen: Each 2x2 block product is computed with 7 scalar products.
    # Total rank = 7 * 7 = 49.
    
    # Let's define the linear forms for the outer Strassen.
    # A is 4x4. Blocks A00, A01, A10, A11.
    # Indices for A: 0..15.
    # A00: 0,1,4,5 (row 0: 0,1; row 1: 4,5)
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # Wait, standard flattening is row-major.
    # A[i][j] is at index i*m + j.
    # m=4.
    # A00: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5.
    # A01: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7.
    # A10: (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13.
    # A11: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15.
    
    # Let's define the 7 outer linear forms L_i and R_i as 4x4 matrices of coefficients.
    # Or better, as 16-dim vectors.
    
    def block_indices(block_row, block_col):
        # Returns the 4 indices of the 2x2 block
        # block_row, block_col in {0, 1}
        start_i = block_row * 2
        start_j = block_col * 2
        return [
            start_i * 4 + start_j,
            start_i * 4 + start_j + 1,
            (start_i + 1) * 4 + start_j,
            (start_i + 1) * 4 + start_j + 1
        ]
    
    # Outer Strassen definitions
    # L_i and R_i are linear combinations of blocks.
    # We represent them as 16-dim vectors.
    
    def zero_vec():
        return [0.0] * 16
    
    def add_vec(v1, v2):
        return [v1[i] + v2[i] for i in range(16)]
    
    def sub_vec(v1, v2):
        return [v1[i] - v2[i] for i in range(16)]
    
    # Define block unit vectors
    # B00, B01, B10, B11
    def block_vec(br, bc):
        v = zero_vec()
        for idx in block_indices(br, bc):
            v[idx] = 1.0
        return v
    
    A00 = block_vec(0, 0)
    A01 = block_vec(0, 1)
    A10 = block_vec(1, 0)
    A11 = block_vec(1, 1)
    
    B00 = block_vec(0, 0)
    B01 = block_vec(0, 1)
    B10 = block_vec(1, 0)
    B11 = block_vec(1, 1)
    
    # Outer L and R
    L1 = add_vec(A00, A11)
    R1 = add_vec(B00, B11)
    
    L2 = add_vec(A10, A11)
    R2 = B00
    
    L3 = A00
    R3 = sub_vec(B01, B11)
    
    L4 = A11
    R4 = sub_vec(B10, B00)
    
    L5 = add_vec(A00, A01)
    R5 = B11
    
    L6 = sub_vec(A10, A00)
    R6 = add_vec(B00, B01)
    
    L7 = sub_vec(A01, A11)
    R7 = add_vec(B10, B11)
    
    outer_L = [L1, L2, L3, L4, L5, L6, L7]
    outer_R = [R1, R2, R3, R4, R5, R6, R7]
    
    # Now for each outer product i, we compute the 2x2 matrix product L_i * R_i.
    # L_i and R_i are 4x4 matrices (linear forms on A and B).
    # We need to extract the 2x2 blocks of L_i and R_i?
    # No, L_i is a 4x4 matrix.
    # Wait, L_i is a linear form on A.
    # The product M_i = L_i * R_i is a 4x4 matrix?
    # No.
    # In Strassen, the inputs are 2x2 matrices of blocks.
    # L_1 = A00 + A11. This is a 2x2 matrix of blocks?
    # No, A00 is a 2x2 matrix.
    # So L_1 is a 2x2 matrix.
    # My `block_vec` creates a 16-dim vector representing the 4x4 matrix A.
    # So L_1 is a 4x4 matrix?
    # A00 is the top-left 2x2 block.
    # So A00 as a 4x4 matrix has non-zeros only in the top-left 2x2.
    # So L_1 = A00 + A11 is a 4x4 matrix with non-zeros in top-left and bottom-right 2x2 blocks.
    # The product M_1 = L_1 * R_1 is a 4x4 matrix.
    # We want to decompose this 4x4 matrix product into 7 scalar products.
    
    # But wait, the standard Strassen algorithm for 4x4 treats the 4x4 matrix as 2x2 blocks of 2x2.
    # The "multiplication" is of these blocks.
    # So M_1 is a 2x2 matrix of blocks, where each block is a 2x2 matrix product.
    # So M_1 is effectively a 4x4 matrix.
    # We need to decompose the computation of this 4x4 matrix into 7 scalar products.
    
    # Let's refine the inner Strassen.
    # The "left" operand for the inner Strassen is the 2x2 matrix L_i.
    # L_i has entries that are linear forms on A.
    # Let L_i = [[l00, l01], [l10, l11]] where each lxy is a 2x2 matrix (linear form on A).
    # Similarly for R_i.
    
    # We need to define l00, l01, l10, l11 for each outer step i.
    
    # For M1: L1 = A00 + A11.
    # L1[0,0] (top-left 2x2 block of the 4x4 matrix L1) is A00[0,0..1,0..1] + A11[0,0..1,0..1]?
    # No.
    # A00 is the top-left block of A.
    # So A00 as a 4x4 matrix is:
    # [a00 a01 0 0]
    # [a10 a11 0 0]
    # [0 0 0 0]
    # [0 0 0 0]
    
    # So L1 = A00 + A11 is:
    # [a00 a01 0 0]     [0 0 0 0]
    # [a10 a11 0 0] +  [0 0 0 0]
    # [0 0 0 0]         [0 0 b00 b01]
    # [0 0 0 0]         [0 0 b10 b11]
    # = [a00 a01 0 0]
    #   [a10 a11 0 0]
    #   [0 0 b00 b01]
    #   [0 0 b10 b11]
    
    # This is NOT a 2x2 matrix of 2x2 blocks in the sense of Strassen?
    # Yes it is.
    # Block (0,0) of L1 is A00.
    # Block (0,1) of L1 is 0.
    # Block (1,0) of L1 is 0.
    # Block (1,1) of L1 is A11.
    
    # So for the inner Strassen, we need the 4 blocks of L1 and R1.
    
    def get_blocks(vec):
        # Returns 4 vectors of size 4, representing the 2x2 blocks of the 4x4 matrix
        # Block 0: 0,1,4,5
        # Block 1: 2,3,6,7
        # Block 2: 8,9,12,13
        # Block 3: 10,11,14,15
        blocks = []
        for br in range(2):
            for bc in range(2):
                b = [0.0] * 4
                for idx in block_indices(br, bc):
                    # Map global index to local index in 2x2 block
                    # local row = (idx // 4) % 2
                    # local col = idx % 4
                    lr = (idx // 4) % 2
                    lc = idx % 4
                    b[lr * 2 + lc] = vec[idx]
                blocks.append(b)
        return blocks

    # Inner Strassen linear forms for 2x2 matrices
    # u1 = [1, 0, 0, 1] (l00 + l11)
    # u2 = [0, 0, 1, 1] (l10 + l11)
    # u3 = [1, 0, 0, 0] (l00)
    # u4 = [0, 0, 0, 1] (l11)
    # u5 = [1, 1, 0, 0] (l00 + l01)
    # u6 = [0, 0, 1, -1] (l10 - l00)
    # u7 = [0, 1, 0, -1] (l01 - l11)
    
    inner_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    
    inner_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Reconstruction for 2x2 C
    # c00 = p1 - p4 + p6
    # c01 = p3 + p5
    # c10 = p2 + p7
    # c11 = p1 - p3 + p4
    
    inner_recon = [
        [1, 0, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, -1, 1, 0, 0, 0]
    ]
    
    # Outer Reconstruction for 4x4 C
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M3 + M4
    
    outer_recon = [
        [1, 0, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, -1, 1, 0, 0, 0]
    ]
    
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    # For each of the two 4x4 by 4x4 multiplications
    for half in range(2):
        # Offset for B and C columns
        # B is 4x8. First 4 cols are for half 0, next 4 for half 1.
        # C is 8x4. First 4 cols are for half 0?
        # C = A * B.
        # C[:, 0:4] = A * B[:, 0:4]
        # C[:, 4:8] = A * B[:, 4:8]
        
        # Indices for B in this half
        # B_col_offset = half * 4
        
        # For each outer product i (0..6)
        for i in range(7):
            L_i = outer_L[i]
            R_i = outer_R[i]
            
            # Get blocks of L_i and R_i
            L_blocks = get_blocks(L_i) # 4 vectors of size 4
            R_blocks = get_blocks(R_i)
            
            # For each inner product j (0..6)
            for j in range(7):
                # Compute scalar linear forms u and v
                # u is a 16-dim vector
                u = [0.0] * 16
                v = [0.0] * 16
                
                # u is combination of L_blocks
                for b_idx in range(4):
                    coeff = inner_u[j][b_idx]
                    if coeff != 0:
                        for k in range(4):
                            # Map local index k to global index
                            # Local block b_idx corresponds to (br, bc)
                            br = b_idx // 2
                            bc = b_idx % 2
                            lr = k // 2
                            lc = k % 2
                            global_idx = (br * 2 + lr) * 4 + (bc * 2 + lc)
                            u[global_idx] += coeff * L_blocks[b_idx][k]
                
                # v is combination of R_blocks
                # Note: R_i is a linear form on B.
                # B is 4x8.
                # For this half, we only care about columns half*4 to half*4+3.
                # The block structure of B is the same.
                for b_idx in range(4):
                    coeff = inner_v[j][b_idx]
                    if coeff != 0:
                        for k in range(4):
                            br = b_idx // 2
                            bc = b_idx % 2
                            lr = k // 2
                            lc = k % 2
                            # Global index in B (4x8)
                            # Row: br * 2 + lr
                            # Col: bc * 2 + lc + half * 4
                            global_row = br * 2 + lr
                            global_col = bc * 2 + lc + half * 4
                            global_idx = global_row * 8 + global_col
                            v[global_idx] += coeff * R_blocks[b_idx][k]
                
                # W coefficients
                # We need to reconstruct C.
                # C is 8x4.
                # C[:, 0:4] and C[:, 4:8].
                # The current product contributes to C[:, half*4 : half*4+4].
                
                # The inner product j contributes to the 2x2 block of M_i.
                # Let's find which 2x2 block of M_i this inner product j contributes to.
                # And with what coefficient.
                
                # The inner Strassen computes M_i = L_i * R_i.
                # M_i is a 4x4 matrix.
                # It is composed of 4 2x2 blocks: M_i00, M_i01, M_i10, M_i11.
                # The inner Strassen products P1..P7 are used to reconstruct these blocks.
                
                # P_j contributes to M_i blocks as defined by inner_recon.
                # inner_recon[0] is for M_i00
                # inner_recon[1] is for M_i01
                # inner_recon[2] is for M_i10
                # inner_recon[3] is for M_i11
                
                # So for each block c_block (0..3) of M_i:
                for c_block in range(4):
                    coeff_inner = inner_recon[c_block][j]
                    if coeff_inner == 0:
                        continue
                    
                    # Now this contribution goes to M_i[c_block].
                    # M_i is the i-th outer product.
                    # M_i contributes to C blocks as defined by outer_recon.
                    
                    # M_i is a 4x4 matrix (2x2 blocks of 2x2).
                    # M_i[c_block] is a 2x2 matrix.
                    # This 2x2 matrix is part of C.
                    
                    # Which block of C does M_i[c_block] correspond to?
                    # M_i is a 4x4 matrix.
                    # C is a 4x4 matrix (for this half).
                    # The outer Strassen reconstructs C from M1..M7.
                    
                    # outer_recon[c_row_block] gives coefficients for M1..M7 to form C[c_row_block].
                    # C is 4x4.
                    # C blocks: C00, C01, C10, C11.
                    
                    # Wait, M_i is a 4x4 matrix.
                    # So M_i has 4 blocks.
                    # The outer Strassen sums M_i to form C.
                    # C = sum_i outer_recon_coeff[i] * M_i.
                    
                    # So M_i contributes to ALL blocks of C?
                    # No.
                    # In Strassen, M1 is a 2x2 matrix of blocks.
                    # C00 = M1 - M4 + M6.
                    # This means the block (0,0) of C is the sum of the block (0,0) of M1, M4, M6.
                    
                    # So M_i[c_block] contributes to C[c_block]?
                    # Yes, because M_i is a 2x2 matrix of blocks, and the Strassen reconstruction is block-wise.
                    
                    # So M_i[c_block] contributes to C[c_block] with coefficient 1?
                    # No, the outer_recon coefficients apply to the whole M_i.
                    
                    # So for a given c_block (0..3) of C:
                    # C[c_block] = sum_i outer_recon[c_block][i] * M_i[c_block].
                    
                    # So the contribution of P_j to C[c_block] is:
                    # outer_recon[c_block][i] * inner_recon[c_block][j] * P_j.
                    
                    coeff_outer = outer_recon[c_block][i]
                    coeff_total = coeff_outer * coeff_inner
                    
                    if coeff_total == 0:
                        continue
                    
                    # Now we need to add this to W.
                    # W has shape (p*n, rank) = (8*4, rank) = (32, rank).
                    # The index in W is k*n + i.
                    # Here, the "i" in the prompt's W[k*n+i] is the row of C.
                    # The "k" is the col of C.
                    
                    # c_block is a 2x2 block of C.
                    # c_block 0: C[0:2, 0:2]
                    # c_block 1: C[0:2, 2:4]
                    # c_block 2: C[2:4, 0:2]
                    # c_block 3: C[2:4, 2:4]
                    
                    # Note: C is 4x4 for this half.
                    # The columns of C for this half are 0..3.
                    # So c_block 0 is C[0:2, 0:2].
                    # c_block 1 is C[0:2, 2:4].
                    # c_block 2 is C[2:4, 0:2].
                    # c_block 3 is C[2:4, 2:4].
                    
                    # We need to fill W for the entries in this block.
                    # The entries are (r, c) where r in 0..1 or 2..3, c in 0..1 or 2..3.
                    
                    br = c_block // 2
                    bc = c_block % 2
                    
                    for lr in range(2):
                        for lc in range(2):
                            row = br * 2 + lr
                            col = bc * 2 + lc
                            
                            # W index: col * n + row
                            # n=4.
                            # But wait, C is 8x4.
                            # The col index in the full C is col + half * 4.
                            # The row index is row.
                            
                            w_col = col + half * 4
                            w_row = row
                            
                            w_idx = w_col * 4 + w_row
                            
                            # We need to create a W vector for this rank term.
                            # The W vector has 32 entries.
                            # Only w_idx is non-zero, with value coeff_total.
                            
                            w_vec = [0.0] * 32
                            w_vec[w_idx] = coeff_total
                            
                            U_cols.append(u)
                            V_cols.append(v)
                            W_cols.append(w_vec)
                            rank += 1
    
    # Construct final matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        U[:, r] = U_cols[r]
        V[:, r] = V_cols[r]
        W[:, r] = W_cols[r]
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
