import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 114 decomposition:
    # 2 * Strassen(4x4x4) [Rank 49 each] + 1 * Schoolbook(4x4x1) [Rank 16]
    # Total Rank = 49 + 49 + 16 = 114
    
    R = 114
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    # Helper to add Strassen 4x4x4 decomposition
    # offset_col: column offset in B/C (0, 4)
    # rank_offset: column offset in U/V/W (0, 49)
    def add_strassen_4x4x4(offset_col, rank_offset):
        # Strassen 4x4x4 is built from 16 Strassen 2x2x2 steps?
        # No, 4x4x4 Strassen is 7^2 = 49.
        # It is constructed by applying Strassen to 2x2 blocks of A, B, C.
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        # We need to map the 7 multiplications of the 2x2 Strassen to the 4x4 blocks.
        # Actually, it's easier to just iterate over the 16 "scalar" Strassen decompositions?
        # No, 4x4 Strassen uses 49 multiplications.
        # The 49 multiplications are formed by taking the 7 Strassen formulas
        # and applying them to the 2x2 blocks, where each "element" is a 2x2 matrix.
        # Wait, that's not how it works.
        # The 4x4 Strassen decomposition is:
        # M1 = (A11+A22)(B11+B22) -> 2x2 * 2x2 -> 4 mults? No, it's a 2x2 matrix product.
        # Rank of 2x2 matrix product is 4 (schoolbook) or 7 (Strassen).
        # If we use Strassen for the inner 2x2 products, we get 7*7=49.
        # So we need to decompose each of the 7 "block multiplications" using 2x2 Strassen.
        
        # Let's generate the 49 terms.
        # Outer Strassen steps (7):
        # 1: (A11+A22) * (B11+B22)
        # 2: (A21+A22) * B11
        # 3: A11 * (B12-B22)
        # 4: A22 * (B21-B11)
        # 5: (A11+A12) * B22
        # 6: (A21-A11) * (B11+B12)
        # 7: (A12-A22) * (B21+B22)
        
        # Inner Strassen steps (7) for each 2x2 * 2x2.
        
        # We need to map the 2x2 indices to the 4x4 indices.
        # Block (bi, bj) of A is at rows bi*2..bi*2+1, cols bj*2..bj*2+1.
        
        # We will iterate over the 7 outer steps and 7 inner steps.
        
        # Outer Step Definitions (coefficients for A blocks and B blocks)
        # A blocks: 00, 01, 10, 11
        # B blocks: 00, 01, 10, 11
        
        outer_steps = [
            # (A_coeff_00, A_coeff_01, A_coeff_10, A_coeff_11, B_coeff_00, B_coeff_01, B_coeff_10, B_coeff_11)
            (1, 0, 0, 1, 1, 0, 0, 1), # M1
            (0, 0, 1, 1, 1, 0, 0, 0), # M2
            (1, 0, 0, 0, 0, 1, 0, -1), # M3
            (0, 0, 0, 1, 0, 0, 1, -1), # M4
            (1, 1, 0, 0, 0, 0, 0, 1), # M5
            (0, 0, 1, -1, 1, 1, 0, 0), # M6
            (0, 1, 0, -1, 0, 0, 1, 1)  # M7
        ]
        
        # Inner Strassen 2x2x2 definitions
        # Returns (U_inner, V_inner, W_inner) for 2x2x2
        # U_inner is 4x7, V_inner is 4x7, W_inner is 4x7
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        
        def get_strassen_2x2x2():
            U2 = np.zeros((4, 7))
            V2 = np.zeros((4, 7))
            W2 = np.zeros((4, 7))
            
            # M1: (A00+A11)(B00+B11)
            U2[0, 0] = 1; U2[3, 0] = 1
            V2[0, 0] = 1; V2[3, 0] = 1
            W2[0, 0] = 1; W2[3, 0] = 1
            
            # M2: (A10+A11)B00
            U2[2, 1] = 1; U2[3, 1] = 1
            V2[0, 1] = 1
            W2[1, 1] = 1; W2[3, 1] = -1 # C10 += M2, C11 -= M2
            
            # M3: A00(B01-B11)
            U2[0, 2] = 1
            V2[1, 2] = 1; V2[3, 2] = -1
            W2[0, 2] = 1; W2[2, 2] = 1 # C01 += M3, C11 += M3
            
            # M4: A11(B10-B00)
            U2[3, 3] = 1
            V2[2, 3] = 1; V2[0, 3] = -1
            W2[1, 3] = 1; W2[0, 3] = 1 # C10 += M4, C00 += M4
            
            # M5: (A00+A01)B11
            U2[0, 4] = 1; U2[1, 4] = 1
            V2[3, 4] = 1
            W2[2, 4] = -1; W2[0, 4] = 1 # C11 -= M5, C01 += M5
            
            # M6: (A10-A00)(B00+B01)
            U2[2, 5] = 1; U2[0, 5] = -1
            V2[0, 5] = 1; V2[1, 5] = 1
            W2[3, 5] = 1 # C11 += M6
            
            # M7: (A01-A11)(B10+B11)
            U2[1, 6] = 1; U2[3, 6] = -1
            V2[2, 6] = 1; V2[3, 6] = 1
            W2[0, 6] = 1 # C00 += M7
            
            return U2, V2, W2
        
        U2, V2, W2 = get_strassen_2x2x2()
        
        for o_idx, (a00, a01, a10, a11, b00, b01, b10, b11) in enumerate(outer_steps):
            # For each inner step i_idx (0..6)
            for i_idx in range(7):
                r = rank_offset + o_idx * 7 + i_idx
                
                # Update U
                # U2[i_idx] has non-zeros at positions (0,1,2,3) corresponding to 2x2 A block.
                # We need to map these to the 4x4 A block positions.
                # The A block for this outer step is a linear combination of 4 2x2 blocks.
                # Wait, the outer step defines the *combination* of blocks.
                # Example M1: A_block = A00 + A11.
                # So for each element in the 2x2 A_block, we add the contributions from A00 and A11.
                
                # U2[i_idx, local_i, local_j] -> Global A[bi*2+local_i, bj*2+local_j]
                
                # We need to iterate over the local 2x2 indices
                for li in range(2):
                    for lj in range(2):
                        val_u = U2[li*2+lj, i_idx]
                        if val_u != 0:
                            # This val_u applies to the "A_block".
                            # A_block[li, lj] = a00*A00[li,lj] + a01*A01[li,lj] + a10*A10[li,lj] + a11*A11[li,lj]
                            # So we add val_u * coeff to the corresponding global U.
                            
                            # A00 is at (0,0)
                            if a00 != 0:
                                U[0*4 + 0*2 + li, r] += val_u * a00 # Row 0*2+li, Col 0*2+lj
                                # Wait, U index is i*m + j. m=4.
                                # i is row, j is col.
                                # A00[li,lj] is at global row li, col lj.
                                U[li*4 + lj, r] += val_u * a00
                            
                            if a01 != 0:
                                # A01 is at (0,1) -> rows 0..1, cols 2..3
                                U[li*4 + (2+lj), r] += val_u * a01
                            
                            if a10 != 0:
                                # A10 is at (1,0) -> rows 2..3, cols 0..1
                                U[(2+li)*4 + lj, r] += val_u * a10
                            
                            if a11 != 0:
                                # A11 is at (1,1) -> rows 2..3, cols 2..3
                                U[(2+li)*4 + (2+lj), r] += val_u * a11
                            
                # Update V
                for lj in range(2): # This is 'j' in 2x2 V, which is 'k' in A? No.
                    # V2 is indexed by [j*2+k].
                    # In 2x2 Strassen, V corresponds to B.
                    # B_block[j,k].
                    for lk in range(2):
                        val_v = V2[lj*2+lk, i_idx]
                        if val_v != 0:
                            # B_block[lj, lk] = b00*B00[lj,lk] + ...
                            # B00 is at (0,0) -> rows 0..1, cols 0..1
                            if b00 != 0:
                                V[(0*2+lj)*9 + (0*2+lk), r] += val_v * b00
                                # V index is j*p + k. p=9.
                                # B00[ lj, lk ] is at global row lj, col lk.
                                V[lj*9 + lk, r] += val_v * b00
                            
                            if b01 != 0:
                                # B01 is at (0,1) -> rows 0..1, cols 2..3
                                V[lj*9 + (2+lk), r] += val_v * b01
                            
                            if b10 != 0:
                                # B10 is at (1,0) -> rows 2..3, cols 0..1
                                V[(2+lj)*9 + lk, r] += val_v * b10
                            
                            if b11 != 0:
                                # B11 is at (1,1) -> rows 2..3, cols 2..3
                                V[(2+lj)*9 + (2+lk), r] += val_v * b11
                            
                # Update W
                # W2 is indexed by [k*2+i].
                # C_block[k,i].
                for lk in range(2):
                    for li in range(2):
                        val_w = W2[lk*2+li, i_idx]
                        if val_w != 0:
                            # C_block[lk, li] contributes to global C.
                            # The outer step determines which C blocks are affected.
                            # M1 -> C00 + C11
                            # M2 -> C10 - C11
                            # M3 -> C01 + C11
                            # M4 -> C00 + C10
                            # M5 -> C01 - C11
                            # M6 -> C11
                            # M7 -> C00
                            
                            # We need a mapping from o_idx to C block coefficients.
                            # c00, c01, c10, c11
                            
                            c_coeffs = {
                                0: (1, 0, 0, 1), # M1
                                1: (0, 0, 1, -1), # M2
                                2: (0, 1, 0, 1), # M3
                                3: (1, 0, 1, 0), # M4
                                4: (0, 1, 0, -1), # M5
                                5: (0, 0, 0, 1), # M6
                                6: (1, 0, 0, 0)  # M7
                            }
                            
                            c00, c01, c10, c11 = c_coeffs[o_idx]
                            
                            # C00 is at (0,0) -> rows 0..1, cols 0..1
                            if c00 != 0:
                                W[(0*2+lk)*4 + (0*2+li), r] += val_w * c00
                                W[(lk)*4 + (li), r] += val_w * c00
                            
                            if c01 != 0:
                                # C01 is at (0,1) -> rows 0..1, cols 2..3
                                W[(lk)*4 + (2+li), r] += val_w * c01
                            
                            if c10 != 0:
                                # C10 is at (1,0) -> rows 2..3, cols 0..1
                                W[(2+lk)*4 + (li), r] += val_w * c10
                            
                            if c11 != 0:
                                # C11 is at (1,1) -> rows 2..3, cols 2..3
                                W[(2+lk)*4 + (2+li), r] += val_w * c11

    # Apply first 4x4x4 Strassen (cols 0-3)
    add_strassen_4x4x4(0, 0)
    
    # Apply second 4x4x4 Strassen (cols 4-7)
    add_strassen_4x4x4(4, 49)
    
    # Apply Schoolbook for 4x4x1 (col 8)
    # Rank 16
    for j in range(4):
        for i in range(4):
            # This is not quite right. Schoolbook for 4x4x1.
            # C_i = sum_j A_ij B_j.
            # We need 16 terms.
            # For each j, we have 4 terms (one for each i)?
            # No, for each (i,j) pair?
            # Term r: U has 1 at (i,j). V has 1 at (j, 8). W has 1 at (8, i).
            # Sum over r: C_i = sum_j A_ij B_j * 1.
            
            # We have 16 ranks available: 49+49 to 49+49+15.
            rank_idx = 98 + i * 4 + j
            U[i*4 + j, rank_idx] = 1
            V[j*9 + 8, rank_idx] = 1
            W[8*4 + i, rank_idx] = 1

    return {'U': U, 'V': V, 'W': W}
