"""
Strategy: Direct Algebraic Construction via Block Decomposition.
We decompose the 5x5 matrix multiplication into a 4x4 core and a 1x1 boundary.
1. The 4x4 core (indices 0-3) is multiplied using the Strassen algorithm.
   - Strassen for 2x2 has rank 7.
   - 4x4 can be viewed as 2x2 blocks of 2x2 matrices.
   - This requires 4 (block positions) * 7 (Strassen rank) = 28? 
   - Wait, Strassen 4x4 rank is 49.
   - The 4x4 matrix is partitioned into four 2x2 blocks: A00, A01, A10, A11.
   - The product C = A*B is computed by applying Strassen to these blocks.
   - Each "multiplication" in the Strassen formula is a 2x2 matrix multiplication.
   - Since we are decomposing the *scalar* tensor, each 2x2 multiplication 
     contributes rank 4 (schoolbook) or can be further decomposed?
     No, the standard rank of 4x4 matrix multiplication is 49.
     This is achieved by applying Strassen to the 4x4 matrix, where the 
     "scalar" multiplications in the Strassen algorithm are actually 
     2x2 matrix multiplications. But wait, the rank of a 2x2 matrix 
     multiplication is 7 (Strassen).
     So, if we use Strassen for the top-level 4x4, we need 7 multiplications 
     of 2x2 matrices. Each 2x2 multiplication has rank 7.
     Total rank = 7 * 7 = 49.
     
     How to construct this?
     We need to map the indices.
     Let the 4x4 indices be i, j, k in {0, 1, 2, 3}.
     We can view i, j, k as pairs (i0, i1) where i = 2*i0 + i1, i0, i1 in {0, 1}.
     The 4x4 multiplication tensor T_4 is the tensor product of the 2x2 
     multiplication tensor T_2 with itself? 
     No, T_4 is not T_2 \otimes T_2. 
     T_4 is the tensor for multiplying 4x4 matrices.
     The Strassen algorithm for 4x4 computes 7 products of 2x2 matrices.
     Let M1, ..., M7 be the 7 intermediate 2x2 products.
     Each Mr is a linear combination of entries of A and B.
     Each entry of Mr is a scalar product of a linear form in A and a linear form in B.
     Since Mr is a 2x2 matrix, its entries are computed by 2x2 matrix multiplication logic?
     No, Mr is just a matrix of values. The computation of Mr involves 
     multiplying 2x2 matrices. 
     So, to compute one 2x2 matrix product (say M1 = A00 * B00 + ...), 
     we need to perform a 2x2 matrix multiplication, which has rank 7.
     Thus, the total rank is 7 (Strassen steps) * 7 (rank of 2x2 mult) = 49.
     
     Construction:
     We define the 49 terms by iterating over the 7 Strassen terms for the 4x4 level,
     and for each, iterating over the 7 Strassen terms for the 2x2 level.
     
     Indices:
     Global indices i, j, k in 0..3.
     Let i = 2*i_hi + i_lo, etc.
     
     The 4x4 Strassen formulas:
     Let A be partitioned into A00, A01, A10, A11 (each 2x2).
     Let B be partitioned into B00, B01, B10, B11 (each 2x2).
     
     S1 = A00 + A11
     S2 = A10 + A11
     S3 = A00 - A01
     S4 = A11 - A10
     S5 = A00 + A01
     S6 = A10 + A01
     S7 = A11 - A01
     
     P1 = S1 * (B00 + B11)
     P2 = S2 * B00
     P3 = S3 * (B01 - B11)
     P4 = S4 * (B10 - B00)
     P5 = S5 * B11
     P6 = (S6 + S7) * B10  -- Wait, standard Strassen:
     P6 = (A10 + A01) * (B01 - B11)? No.
     Let's use the standard 2x2 Strassen formulas for the blocks.
     
     C00 = P1 + P4 - P5 + P7
     C01 = P3 + P5
     C10 = P2 + P4
     C11 = P1 - P2 + P3 + P6
     
     Where:
     P1 = (A00+A11)(B00+B11)
     P2 = (A10+A11)B00
     P3 = A00(B01-B11)
     P4 = A11(B10-B00)
     P5 = (A00+A01)B11
     P6 = (A10-A11)(B01+B11)
     P7 = (A01-A11)(B00+B10)
     
     Each Pr is a 2x2 matrix product.
     To compute Pr = X * Y where X and Y are 2x2 matrices (which are linear combinations of A/B blocks),
     we use the rank-7 decomposition of 2x2 multiplication.
     
     So, for each of the 7 P's, we have 7 scalar multiplications.
     Total 49 scalar multiplications for the 4x4 core.
     
     The 1x1 boundary (index 4):
     A is 5x5. Blocks:
     A44 (4x4), A41 (4x1), A14 (1x4), A11 (1x1).
     B44 (4x4), B41 (4x1), B14 (1x4), B11 (1x1).
     
     C44 = A44 B44 + A41 B14
     C41 = A44 B41 + A41 B11
     C14 = A14 B44 + A11 B14
     C11 = A14 B41 + A11 B11
     
     We already handled A44 B44 with rank 49.
     We need to add:
     1. A41 B14: (4x1) * (1x4) -> 4x4 result.
        Terms: A[i,4] * B[4,k] for i,k in 0..3.
        This is rank 16 (schoolbook).
     2. A44 B41: (4x4) * (4x1) -> 4x1 result.
        Terms: Sum_j A[i,j] B[j,4] for i in 0..3.
        This is rank 16 (schoolbook).
     3. A14 B44: (1x4) * (4x4) -> 1x4 result.
        Terms: Sum_j A[4,j] B[j,k] for k in 0..3.
        This is rank 16 (schoolbook).
     4. A11 B11: (1x1) * (1x1) -> 1x1 result.
        Term: A[4,4] * B[4,4].
        This is rank 1.
     5. A41 B11: (4x1) * (1x1) -> 4x1 result.
        Terms: A[i,4] * B[4,4] for i in 0..3.
        This is rank 4.
     6. A14 B41: (1x4) * (4x1) -> 1x1 result.
        Term: Sum_j A[4,j] B[j,4].
        This is rank 4.
        
     Wait, let's list all cross terms carefully.
     C = A B.
     C[i,k] = sum_j A[i,j] B[j,k].
     
     Indices 0..3 are "core", 4 is "boundary".
     
     Case 1: i in 0..3, k in 0..3 (C44)
     C[i,k] = sum_{j=0..3} A[i,j] B[j,k] + A[i,4] B[4,k]
     Part 1 (Core-Core): Handled by 4x4 Strassen (Rank 49).
     Part 2 (Core-Boundary): A[i,4] B[4,k].
     For each i in 0..3, k in 0..3, we have a term A[i,4] B[4,k].
     This is a rank-1 term for each (i,k). Total 4*4 = 16 terms.
     
     Case 2: i in 0..3, k = 4 (C41)
     C[i,4] = sum_{j=0..3} A[i,j] B[j,4] + A[i,4] B[4,4]
     Part 1 (Core-Boundary): Sum_{j=0..3} A[i,j] B[j,4].
     For each i in 0..3, this is a sum of 4 terms.
     We can decompose this as 4 separate rank-1 terms for each j.
     Term: A[i,j] B[j,4].
     For fixed i, j, this contributes to C[i,4].
     Total terms: 4 (i) * 4 (j) = 16 terms.
     Part 2 (Boundary-Boundary): A[i,4] B[4,4].
     For each i in 0..3, term A[i,4] B[4,4].
     Total 4 terms.
     
     Case 3: i = 4, k in 0..3 (C14)
     C[4,k] = sum_{j=0..3} A[4,j] B[j,k] + A[4,4] B[4,k]
     Part 1 (Boundary-Core): Sum_{j=0..3} A[4,j] B[j,k].
     For each k in 0..3, sum over j.
     Terms: A[4,j] B[j,k].
     Total 4 (k) * 4 (j) = 16 terms.
     Part 2 (Boundary-Boundary): A[4,4] B[4,k].
     For each k in 0..3, term A[4,4] B[4,k].
     Total 4 terms.
     
     Case 4: i = 4, k = 4 (C11)
     C[4,4] = sum_{j=0..3} A[4,j] B[j,4] + A[4,4] B[4,4]
     Part 1 (Boundary-Core): Sum_{j=0..3} A[4,j] B[j,4].
     Terms: A[4,j] B[j,4].
     Total 4 terms.
     Part 2 (Boundary-Boundary): A[4,4] B[4,4].
     Total 1 term.
     
     Total Rank:
     Core-Core: 49
     Core-Boundary (C44 part): 16
     Core-Boundary (C41 part 1): 16
     Core-Boundary (C41 part 2): 4
     Boundary-Core (C14 part 1): 16
     Boundary-Core (C14 part 2): 4
     Boundary-Core (C11 part 1): 4
     Boundary-Boundary (C11 part 2): 1
     
     Sum: 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110.
     
     Wait, can we optimize the boundary terms?
     The terms A[i,4] B[4,k] are distinct rank-1 tensors.
     A[i,j] B[j,4] are distinct.
     A[4,j] B[j,k] are distinct.
     A[4,4] B[4,4] is distinct.
     
     Is it possible to share computations?
     In standard matrix multiplication, no, because the indices are distinct.
     However, we are just summing rank-1 tensors.
     The rank of the sum is at most the sum of ranks.
     It could be lower if there are linear dependencies, but with integer coefficients and distinct support sets, it's likely the sum.
     
     So rank 110 is a valid upper bound.
     This is significantly better than 125.
     
     Let's implement this.
"""

import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0
    
    n, m, p = 5, 5, 5
    
    # We will build the U, V, W matrices as lists of lists.
    # We don't know the final rank until we finish, so we'll use a list of terms.
    # Each term is a tuple (u_dict, v_dict, w_dict) where keys are flat indices.
    # u_dict: {i*m + j: coeff}
    # v_dict: {j*p + k: coeff}
    # w_dict: {k*n + i: coeff}
    
    terms = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs, v_coeffs, w_coeffs are dicts {flat_index: value}
        terms.append((u_coeffs, v_coeffs, w_coeffs))
        
    # Helper to get flat indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # --- Part 1: 4x4 Core (Indices 0-3) using Strassen ---
    
    # Strassen 2x2 terms definition
    # Returns list of (u_coeffs, v_coeffs, w_coeffs) for 2x2 mult
    # Indices i,j,k in 0..1
    def strassen_2x2_terms(offset_i=0, offset_j=0, offset_k=0):
        # This function generates the 7 terms for a 2x2 multiplication
        # embedded at specific offsets in the 5x5 grid.
        # But wait, the 4x4 Strassen uses 2x2 blocks.
        # The "multiplication" in the 4x4 Strassen formula is a 2x2 matrix mult.
        # So we need to generate the 7 terms for a 2x2 mult, 
        # but the U, V, W coefficients for these terms are themselves linear combinations 
        # of the 2x2 block entries.
        
        # Let's define the 7 Strassen terms for 2x2 explicitly.
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = [[c11, c12], [c21, c22]]
        
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-d)(f+h)
        # M7 = (b-d)(e+g)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to the global 5x5 indices.
        # The 2x2 block is at (offset_i, offset_j) for A, (offset_j, offset_k) for B?
        # No, the 4x4 block is 0..3.
        # We partition 0..3 into 2x2 blocks:
        # Block (0,0): i in 0,1; j in 0,1
        # Block (0,1): i in 0,1; j in 2,3
        # Block (1,0): i in 2,3; j in 0,1
        # Block (1,1): i in 2,3; j in 2,3
        
        # The Strassen algorithm for 4x4 computes 7 products of 2x2 matrices.
        # Let's call the 2x2 blocks of A as A00, A01, A10, A11.
        # A00 is i in 0,1; j in 0,1.
        # A01 is i in 0,1; j in 2,3.
        # A10 is i in 2,3; j in 0,1.
        # A11 is i in 2,3; j in 2,3.
        
        # Similarly for B.
        
        # The 7 products are:
        # P1 = (A00+A11) * (B00+B11)
        # P2 = (A10+A11) * B00
        # P3 = A00 * (B01-B11)
        # P4 = A11 * (B10-B00)
        # P5 = (A00+A01) * B11
        # P6 = (A10-A11) * (B01+B11)
        # P7 = (A01-A11) * (B00+B10)
        
        # Each P is a 2x2 matrix.
        # To compute P = X * Y (where X, Y are 2x2), we use the rank-7 decomposition of 2x2 mult.
        # So for each P, we need to add 7 terms to the global list.
        # The coefficients for these terms will be linear combinations of the A and B entries.
        
        # Let's define a function to add a term for a 2x2 multiplication 
        # where the input matrices X and Y are linear combinations of the base A and B blocks.
        
        def add_2x2_mult_term(x_coeffs, y_coeffs, z_coeffs):
            # x_coeffs: dict {(bi, bj): coeff} for the X matrix (2x2)
            # y_coeffs: dict {(bj, bk): coeff} for the Y matrix (2x2)
            # z_coeffs: dict {(bk, bi): coeff} for the Z matrix (2x2 result)
            # Here bi, bj, bk are local 0,1 indices within the 2x2 block.
            # But we need to map to global indices.
            # The 2x2 block position is determined by the "block index" (0 or 1) in the 4x4 partition.
            # Let's pass the block offsets explicitly.
            pass
            
        # This nested approach is getting complicated. Let's flatten it.
        # We will iterate over the 7 Strassen steps for the 4x4 level.
        # For each step, we define the linear forms for X and Y (2x2 matrices).
        # Then we iterate over the 7 Strassen steps for the 2x2 level.
        # For each of those, we define the scalar multiplications.
        
        # Global Index Mapping:
        # i_global = 2 * i_block + i_local
        # j_global = 2 * j_block + j_local
        # k_global = 2 * k_block + k_local
        
        # The 4x4 Strassen steps define linear combinations of the 2x2 blocks.
        # Let's define the 7 steps.
        # Each step involves multiplying two 2x2 matrices X and Y.
        # X and Y are linear combinations of A's blocks.
        
        # Step 1: P1 = (A00+A11) * (B00+B11)
        # X = A00 + A11.
        # X is a 2x2 matrix. Its entries are:
        # X[i_local, j_local] = A[2*0+i_local, 2*0+j_local] + A[2*1+i_local, 2*1+j_local]
        # Y = B00 + B11.
        # Y[i_local, j_local] = B[2*0+i_local, 2*0+j_local] + B[2*1+i_local, 2*1+j_local]
        # Result Z is added to C00 and C11.
        # C00 is the top-left 2x2 block of C. Indices i in 0,1; k in 0,1.
        # C11 is the bottom-right 2x2 block of C. Indices i in 2,3; k in 2,3.
        # So Z contributes to C[2*0+i_local, 2*0+k_local] and C[2*1+i_local, 2*1+k_local].
        
        # We need to add the rank-7 decomposition of Z = X * Y.
        # The rank-7 decomposition of 2x2 mult produces 7 scalar terms.
        # Each scalar term is of the form:
        # (linear_comb_of_X) * (linear_comb_of_Y) * (linear_comb_of_Z_positions)
        
        # Let's define the 7 scalar terms for 2x2 mult.
        # Let X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]], Z = [[z00, z01], [z10, z11]]
        
        # Term 1: (x00+x11)(y00+y11) -> contributes to z00 (+1) and z11 (+1)
        # Term 2: (x10+x11)y00 -> contributes to z10 (+1)
        # Term 3: x00(y01-y11) -> contributes to z01 (+1) and z11 (+1)
        # Term 4: x11(y10-y00) -> contributes to z00 (+1) and z10 (+1)
        # Term 5: (x00+x01)y11 -> contributes to z00 (-1) and z01 (+1)
        # Term 6: (x10-x11)(y01+y11) -> contributes to z11 (+1)
        # Term 7: (x01-x11)(y00+y10) -> contributes to z00 (+1)
        
        # We will iterate over these 7 terms for each of the 7 4x4 Strassen steps.
        
        def add_4x4_strassen_term(step_id, x_block_coeffs, y_block_coeffs, z_block_coeffs):
            # x_block_coeffs: dict {(bi, bj): coeff} where bi,bj in 0,1.
            # This defines X as a linear combination of the 2x2 blocks of A.
            # Actually, X is a 2x2 matrix whose entries are linear combinations of A's entries.
            # Let's define X entries directly as dicts of global (i,j) -> coeff.
            # But X is 2x2. So we have 4 entries.
            # Let's say X_local[i_l, j_l] = sum_{i_g, j_g} coeff[i_g, j_g] A[i_g, j_g]
            
            # We will generate the 7 scalar terms for X*Y.
            # For each scalar term, we have:
            # u_coeffs: dict {i_g*m+j_g: val}
            # v_coeffs: dict {j_g*p+k_g: val}
            # w_coeffs: dict {k_g*n+i_g: val}
            
            # The 7 scalar terms for 2x2 mult are defined in terms of X_local and Y_local entries.
            # Let's list them:
            
            # 1. u: x00+x11, v: y00+y11, w: z00+1, z11+1
            # 2. u: x10+x11, v: y00, w: z10+1
            # 3. u: x00, v: y01-y11, w: z01+1, z11+1
            # 4. u: x11, v: y10-y00, w: z00+1, z10+1
            # 5. u: x00+x01, v: y11, w: z00-1, z01+1
            # 6. u: x10-x11, v: y01+y11, w: z11+1
            # 7. u: x01-x11, v: y00+y10, w: z00+1
            
            # We need to expand these local variables to global indices.
            # X_local[i_l, j_l] is a linear combination of A entries.
            # We represent X_local as a dict {(i_l, j_l): {global_ij: coeff}}
            # Or simpler: we just pass the global coefficients for each local entry.
            
            # Let's restructure:
            # x_entries: list of 4 dicts. x_entries[i_l*m+j_l] = {global_u_idx: coeff}
            # y_entries: list of 4 dicts. y_entries[j_l*p+k_l] = {global_v_idx: coeff}
            # z_entries: list of 4 dicts. z_entries[k_l*n+i_l] = {global_w_idx: coeff}
            
            # For Term 1:
            # u_part = x_entries[0] + x_entries[3] (x00 + x11)
            # v_part = y_entries[0] + y_entries[3] (y00 + y11)
            # w_part = z_entries[0] + z_entries[3] (z00 + z11)
            
            def merge_dicts(d1, d2, sign=1):
                res = dict(d1)
                for k, v in d2.items():
                    res[k] = res.get(k, 0) + sign * v
                return res
            
            # Term 1
            u1 = merge_dicts(x_entries[0], x_entries[3])
            v1 = merge_dicts(y_entries[0], y_entries[3])
            w1 = merge_dicts(z_entries[0], z_entries[3])
            add_term(u1, v1, w1)
            
            # Term 2
            u2 = merge_dicts(x_entries[2], x_entries[3]) # x10 + x11
            v2 = dict(y_entries[0]) # y00
            w2 = dict(z_entries[2]) # z10
            add_term(u2, v2, w2)
            
            # Term 3
            u3 = dict(x_entries[0]) # x00
            v3 = merge_dicts(y_entries[1], y_entries[3], sign=-1) # y01 - y11
            w3 = merge_dicts(z_entries[1], z_entries[3]) # z01 + z11
            add_term(u3, v3, w3)
            
            # Term 4
            u4 = dict(x_entries[3]) # x11
            v4 = merge_dicts(y_entries[2], y_entries[0], sign=-1) # y10 - y00
            w4 = merge_dicts(z_entries[0], z_entries[2]) # z00 + z10
            add_term(u4, v4, w4)
            
            # Term 5
            u5 = merge_dicts(x_entries[0], x_entries[1]) # x00 + x01
            v5 = dict(y_entries[3]) # y11
            w5 = merge_dicts(z_entries[0], z_entries[1], sign=-1) # z00 - z01? 
            # Wait, C00 = M1 + M4 - M5 + M7. So M5 contributes -1 to C00.
            # C01 = M3 + M5. So M5 contributes +1 to C01.
            # So w5 should be z00 with -1, z01 with +1.
            w5 = {**z_entries[0], **z_entries[1]}
            for k in w5:
                if k in z_entries[0]:
                    w5[k] = -z_entries[0][k] # This is wrong if keys overlap?
                    # z_entries[0] is for z00 (local k=0, i=0).
                    # z_entries[1] is for z01 (local k=1, i=0).
                    # They are distinct global indices.
                    # So we just need to negate the coeffs in z_entries[0].
            # Let's redo merge with signs.
            def merge_signed(d1, s1, d2, s2):
                res = {}
                for k, v in d1.items(): res[k] = s1 * v
                for k, v in d2.items(): res[k] = res.get(k, 0) + s2 * v
                return res
            
            w5 = merge_signed(z_entries[0], -1, z_entries[1], 1)
            add_term(u5, v5, w5)
            
            # Term 6
            u6 = merge_signed(x_entries[2], 1, x_entries[3], -1) # x10 - x11
            v6 = merge_dicts(y_entries[1], y_entries[3]) # y01 + y11
            w6 = dict(z_entries[3]) # z11
            add_term(u6, v6, w6)
            
            # Term 7
            u7 = merge_signed(x_entries[1], 1, x_entries[3], -1) # x01 - x11
            v7 = merge_dicts(y_entries[0], y_entries[2]) # y00 + y10
            w7 = dict(z_entries[0]) # z00
            add_term(u7, v7, w7)
            
        # Now we define the 7 4x4 Strassen steps.
        
        # Helper to create local entries for a 2x2 block of A.
        # block (bi, bj) corresponds to global i in 2*bi..2*bi+1, j in 2*bj..2*bj+1.
        def get_a_block(bi, bj):
            # Returns list of 4 dicts for local (0,0), (0,1), (1,0), (1,1)
            entries = []
            for il in range(2):
                for jl in range(2):
                    ig = 2*bi + il
                    jg = 2*bj + jl
                    entries.append({u_idx(ig, jg): 1})
            return entries
            
        def get_b_block(bj, bk):
            entries = []
            for jl in range(2):
                for kl in range(2):
                    jg = 2*bj + jl
                    kg = 2*bk + kl
                    entries.append({v_idx(jg, kg): 1})
            return entries
            
        def get_c_block(bk, bi):
            entries = []
            for kl in range(2):
                for il in range(2):
                    kg = 2*bk + kl
                    ig = 2*bi + il
                    entries.append({w_idx(kg, ig): 1})
            return entries
            
        # Step 1: P1 = (A00+A11) * (B00+B11)
        # X = A00 + A11
        x1 = [merge_dicts(get_a_block(0,0)[i], get_a_block(1,1)[i]) for i in range(4)]
        # Y = B00 + B11
        y1 = [merge_dicts(get_b_block(0,0)[i], get_b_block(1,1)[i]) for i in range(4)]
        # Z = C00 + C11
        z1 = [merge_dicts(get_c_block(0,0)[i], get_c_block(1,1)[i]) for i in range(4)]
        add_4x4_strassen_term(1, x1, y1, z1)
        
        # Step 2: P2 = (A10+A11) * B00
        x2 = [merge_dicts(get_a_block(1,0)[i], get_a_block(1,1)[i]) for i in range(4)]
        y2 = get_b_block(0,0)
        z2 = get_c_block(0,1) # C10 is block (1,0) in C? 
        # C10 corresponds to i in 2,3 (block 1), k in 0,1 (block 0).
        # So C block (1,0).
        z2 = get_c_block(0,1) # Wait, get_c_block(bk, bi). C10 is bk=0, bi=1.
        add_4x4_strassen_term(2, x2, y2, z2)
        
        # Step 3: P3 = A00 * (B01-B11)
        x3 = get_a_block(0,0)
        y3 = [merge_signed(get_b_block(0,1)[i], 1, get_b_block(1,1)[i], -1) for i in range(4)]
        z3 = get_c_block(1,0) # C01 is bk=1, bi=0.
        add_4x4_strassen_term(3, x3, y3, z3)
        
        # Step 4: P4 = A11 * (B10-B00)
        x4 = get_a_block(1,1)
        y4 = [merge_signed(get_b_block(1,0)[i], 1, get_b_block(0,0)[i], -1) for i in range(4)]
        z4 = [merge_dicts(get_c_block(0,0)[i], get_c_block(0,1)[i]) for i in range(4)] # C00 + C10?
        # C00 = M1 + M4 - M5 + M7. M4 contributes to C00.
        # C10 = M2 + M4. M4 contributes to C10.
        # So Z4 is C00 (block 0,0) and C10 (block 0,1)?
        # No, C00 is block (0,0) of C. C10 is block (0,1) of C?
        # C is 4x4.
        # C00 (top-left) is i in 0,1; k in 0,1. Block (0,0).
        # C01 (top-right) is i in 0,1; k in 2,3. Block (0,1).
        # C10 (bottom-left) is i in 2,3; k in 0,1. Block (1,0).
        # C11 (bottom-right) is i in 2,3; k in 2,3. Block (1,1).
        
        # My previous z assignments:
        # Step 1: Z = C00 + C11. Blocks (0,0) and (1,1).
        # Step 2: Z = C10. Block (1,0).
        # Step 3: Z = C01 + C11? 
        # C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
        # So M3 contributes to C01 and C11.
        # So Z3 should be C01 (block 0,1) and C11 (block 1,1).
        z3 = [merge_dicts(get_c_block(1,0)[i], get_c_block(1,1)[i]) for i in range(4)] # Block (0,1) and (1,1).
        # get_c_block(bk, bi). C01 is bk=1, bi=0. C11 is bk=1, bi=1.
        
        # Step 4: M4 contributes to C00 and C10.
        # C00 is block (0,0). C10 is block (1,0).
        z4 = [merge_dicts(get_c_block(0,0)[i], get_c_block(0,1)[i]) for i in range(4)] # Block (0,0) and (1,0).
        # get_c_block(bk, bi). C00 is bk=0, bi=0. C10 is bk=0, bi=1.
        
        add_4x4_strassen_term(4, x4, y4, z4)
        
        # Step 5: P5 = (A00+A01) * B11
        x5 = [merge_dicts(get_a_block(0,0)[i], get_a_block(0,1)[i]) for i in range(4)]
        y5 = get_b_block(1,1)
        # M5 contributes to C00 (-1) and C01 (+1).
        # C00 is block (0,0). C01 is block (0,1).
        z5 = [merge_signed(get_c_block(0,0)[i], -1, get_c_block(1,0)[i], 1) for i in range(4)]
        # get_c_block(bk, bi). C00 is bk=0, bi=0. C01 is bk=1, bi=0.
        add_4x4_strassen_term(5, x5, y5, z5)
        
        # Step 6: P6 = (A10-A11) * (B01+B11)
        x6 = [merge_signed(get_a_block(1,0)[i], 1, get_a_block(1,1)[i], -1) for i in range(4)]
        y6 = [merge_dicts(get_b_block(0,1)[i], get_b_block(1,1)[i]) for i in range(4)]
        # M6 contributes to C11.
        # C11 is block (1,1).
        z6 = get_c_block(1,1)
        add_4x4_strassen_term(6, x6, y6, z6)
        
        # Step 7: P7 = (A01-A11) * (B00+B10)
        x7 = [merge_signed(get_a_block(0,1)[i], 1, get_a_block(1,1)[i], -1) for i in range(4)]
        y7 = [merge_dicts(get_b_block(0,0)[i], get_b_block(1,0)[i]) for i in range(4)]
        # M7 contributes to C00.
        # C00 is block (0,0).
        z7 = get_c_block(0,0)
        add_4x4_strassen_term(7, x7, y7, z7)
        
    # --- Part 2: Boundary Terms ---
    
    # 1. A41 B14 (Core-Boundary in C44)
    # A[i,4] B[4,k] for i,k in 0..3
    for i in range(4):
        for k in range(4):
            u = {u_idx(i, 4): 1}
            v = {v_idx(4, k): 1}
            w = {w_idx(k, i): 1}
            add_term(u, v, w)
            
    # 2. A44 B41 (Core-Boundary in C41)
    # Sum_j A[i,j] B[j,4] for i in 0..3
    for i in range(4):
        for j in range(4):
            u = {u_idx(i, j): 1}
            v = {v_idx(j, 4): 1}
            w = {w_idx(4, i): 1}
            add_term(u, v, w)
            
    # 3. A14 B44 (Boundary-Core in C14)
    # Sum_j A[4,j] B[j,k] for k in 0..3
    for k in range(4):
        for j in range(4):
            u = {u_idx(4, j): 1}
            v = {v_idx(j, k): 1}
            w = {w_idx(k, 4): 1}
            add_term(u, v, w)
            
    # 4. A11 B11 (Boundary-Boundary in C11)
    u = {u_idx(4, 4): 1}
    v = {v_idx(4, 4): 1}
    w = {w_idx(4, 4): 1}
    add_term(u, v, w)
    
    # 5. A41 B11 (Core-Boundary in C41)
    # A[i,4] B[4,4] for i in 0..3
    for i in range(4):
        u = {u_idx(i, 4): 1}
        v = {v_idx(4, 4): 1}
        w = {w_idx(4, i): 1}
        add_term(u, v, w)
        
    # 6. A14 B41 (Boundary-Core in C11)
    # Sum_j A[4,j] B[j,4]
    for j in range(4):
        u = {u_idx(4, j): 1}
        v = {v_idx(j, 4): 1}
        w = {w_idx(4, 4): 1}
        add_term(u, v, w)

    # Convert terms to U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
