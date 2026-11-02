"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy:
    # 1. Split the 4x4x6 multiplication into a 4x4x4 part and a 4x4x2 part.
    # 2. Use the known rank-49 Strassen decomposition for 4x4x4.
    # 3. Use the known rank-28 Strassen decomposition for 4x4x2.
    # 4. Total rank = 49 + 28 = 77.
    # 5. Verify correctness using numpy. If it fails, fallback to schoolbook (rank 96).
    
    # Helper to build the decomposition for a general n x m x p using Strassen logic
    # Specifically for 4x4x4 and 4x4x2.
    
    # The Strassen 2x2x2 terms (rank 7)
    # Indices: A[2x2], B[2x2], C[2x2]
    # Flattened vectors of length 4.
    # Order: (0,0), (0,1), (1,0), (1,1)
    
    # We define the 7 rank-1 tensors for 2x2x2 multiplication.
    # T_r = u_r \otimes v_r \otimes w_r
    # C = sum_r w_r * (u_r^T A v_r) ? No.
    # C_{ij} = sum_k A_{ik} B_{kj}
    # In tensor notation: C = sum_r (U_r \otimes V_r \otimes W_r) where
    # C_{ij} = sum_r U_{ri} V_{rj} W_{rk} ... wait, dimensions.
    
    # Let's use the explicit coefficients from the Strassen algorithm.
    # For 2x2 matrices A, B, C:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to express this as a sum of rank-1 terms in the tensor M_{ijk} = A_{ik} B_{kj}.
    # Actually, the standard decomposition is:
    # C_{ij} = sum_{r=1}^7 c_{ij}^{(r)} M_r
    # Where M_r are the 7 scalar multiplications.
    # But M_r is a product of two linear forms.
    # So M_r = (sum a_{ik}^{(r)} A_{ik}) * (sum b_{kj}^{(r)} B_{kj})
    # Then C_{ij} = sum_r (sum a_{ik}^{(r)} A_{ik}) (sum b_{kj}^{(r)} B_{kj}) c_{ij}^{(r)}
    # This is a sum of 7 terms, each of which is a product of 3 linear forms (one for A, one for B, one for C).
    # So it is a rank 7 decomposition.
    
    # Let's extract the coefficients.
    # Term 1:
    # A-part: A00 + A11 -> [1, 0, 0, 1]
    # B-part: B00 + B11 -> [1, 0, 0, 1]
    # C-part: C00 + C11 (with coeff 1) -> Wait, C00 gets +1, C11 gets -1?
    # Let's check C00: M1 contributes +1.
    # Let's check C11: M1 contributes -1.
    # So C-part vector for Term 1: [1, 0, 0, -1] (indices 00, 01, 10, 11)
    
    # Term 2:
    # A-part: A10 + A11 -> [0, 0, 1, 1]
    # B-part: B00 -> [1, 0, 0, 0]
    # C-part: C10 gets +1, C11 gets -1. -> [0, 0, 1, -1]
    
    # Term 3:
    # A-part: A00 -> [1, 0, 0, 0]
    # B-part: B01 - B11 -> [0, 1, 0, -1]
    # C-part: C01 gets +1, C11 gets +1. -> [0, 1, 0, 1]
    
    # Term 4:
    # A-part: A11 -> [0, 0, 0, 1]
    # B-part: B10 - B00 -> [0, -1, 1, 0]
    # C-part: C00 gets +1, C10 gets +1. -> [1, 0, 1, 0]
    
    # Term 5:
    # A-part: A00 + A01 -> [1, 1, 0, 0]
    # B-part: B11 -> [0, 0, 0, 1]
    # C-part: C00 gets -1, C01 gets +1. -> [-1, 1, 0, 0]
    
    # Term 6:
    # A-part: A10 - A00 -> [-1, 0, 1, 0]
    # B-part: B00 + B01 -> [1, 1, 0, 0]
    # C-part: C11 gets +1. -> [0, 0, 0, 1]
    
    # Term 7:
    # A-part: A01 - A11 -> [0, 1, 0, -1]
    # B-part: B10 + B11 -> [0, 0, 1, 1]
    # C-part: C00 gets +1, C11 gets -1. -> [1, 0, 0, -1]
    
    # Let's verify Term 1 contribution to C00:
    # A00B00 + A00B11 + A11B00 + A11B11.
    # This is not a rank 1 term in the tensor M_{ijk} = A_{ik}B_{kj} directly?
    # No, the tensor is M_{ijk} = A_{ik} B_{kj}.
    # The term is (A00+A11)(B00+B11) * (C00 - C11).
    # This means the rank-1 tensor is:
    # T1_{ijk} = (A00+A11)_{ik} (B00+B11)_{kj} (C00-C11)_{ji}?
    # No, the output is C.
    # C_{ij} = sum_k A_{ik} B_{kj}.
    # The decomposition is C = sum_r M_r \otimes K_r ?
    # Actually, the rank of the tensor T_{ijk} = \delta_{ik}\delta_{kj} is n.
    # Here we are decomposing the multiplication map.
    # The map is A, B -> C.
    # C_{ij} = sum_k A_{ik} B_{kj}.
    # This is a 3-tensor T_{ijk} where T_{ijk} = 1 if k=i and j=k? No.
    # T_{i j k} = A_{i k} B_{k j}.
    # So T_{i j k} = \delta_{i, k} \delta_{k, j} ? No.
    # The tensor of multiplication is defined by its action on A and B.
    # T = sum_{i,j,k} A_{ik} B_{kj} E_{ij} \otimes E_{ki} \otimes E_{jk}?
    # Let's stick to the vector form.
    # C = sum_r (u_r \cdot A) (v_r \cdot B) w_r
    # Where u_r, v_r, w_r are vectors.
    # But A and B are matrices.
    # So u_r is a matrix?
    # Yes, the rank-1 terms are of the form A^r \otimes B^r \otimes C^r.
    # C = sum_r (A \cdot A^r) (B \cdot B^r) C^r ?
    # No, C = sum_r (A \circ A^r) (B \circ B^r) C^r ?
    # The standard definition:
    # C_{ij} = sum_r (sum_{a,b} A_{ab} U_{ab}^{(r)}) (sum_{c,d} B_{cd} V_{cd}^{(r)}) W_{ij}^{(r)}
    # This is a rank R decomposition.
    
    # For 2x2x2, R=7.
    # The U, V, W matrices are 4x7.
    
    U2x2 = np.array([
        [1, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, -1, 1, 0, 0, -1]
    ])
    # Check indices: 0:00, 1:01, 2:10, 3:11
    # Term 1: A00, A11 -> [1, 0, 0, 1]
    # Term 2: A10, A11 -> [0, 0, 1, 1]
    # Term 3: A00 -> [1, 0, 0, 0]
    # Term 4: A11 -> [0, 0, 0, 1]
    # Term 5: A00, A01 -> [1, 1, 0, 0]
    # Term 6: A10, A00 -> [1, 0, 1, 0] with signs? A10-A00 -> [-1, 0, 1, 0]
    # Term 7: A01, A11 -> [0, 1, 0, -1]
    
    # Let's re-verify U2x2 rows.
    # Row 0 (Term 1): [1, 0, 0, 1] -> A00+A11. Correct.
    # Row 1 (Term 2): [0, 0, 1, 1] -> A10+A11. Correct.
    # Row 2 (Term 3): [1, 0, 0, 0] -> A00. Correct.
    # Row 3 (Term 4): [0, 0, 0, 1] -> A11. Correct.
    # Row 4 (Term 5): [1, 1, 0, 0] -> A00+A01. Correct.
    # Row 5 (Term 6): [-1, 0, 1, 0] -> A10-A00. Correct.
    # Row 6 (Term 7): [0, 1, 0, -1] -> A01-A11. Correct.
    
    V2x2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    # Term 1: B00+B11 -> [1, 0, 0, 1]
    # Term 2: B00 -> [1, 0, 0, 0]
    # Term 3: B01-B11 -> [0, 1, 0, -1]
    # Term 4: B10-B00 -> [0, -1, 1, 0]
    # Term 5: B11 -> [0, 0, 0, 1]
    # Term 6: B00+B01 -> [1, 1, 0, 0]
    # Term 7: B10+B11 -> [0, 0, 1, 1]
    
    # Check V2x2 rows.
    # Row 0: [1, 0, 0, 1]. Correct.
    # Row 1: [1, 0, 0, 0]. Correct.
    # Row 2: [0, 1, 0, -1]. Correct.
    # Row 3: [0, -1, 1, 0]. Correct.
    # Row 4: [0, 0, 0, 1]. Correct.
    # Row 5: [1, 1, 0, 0]. Correct.
    # Row 6: [0, 0, 1, 1]. Correct.
    
    W2x2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [-1, -1, 1, 1, 0, 1, -1]
    ])
    # Term 1: C00+1, C11-1 -> [1, 0, 0, -1]
    # Term 2: C10+1, C11-1 -> [0, 0, 1, -1]
    # Term 3: C01+1, C11+1 -> [0, 1, 0, 1]
    # Term 4: C00+1, C10+1 -> [1, 0, 1, 0]
    # Term 5: C00-1, C01+1 -> [-1, 1, 0, 0]
    # Term 6: C11+1 -> [0, 0, 0, 1]
    # Term 7: C00+1, C11-1 -> [1, 0, 0, -1]
    
    # Check W2x2 rows.
    # Row 0: [1, 0, 0, -1]. Correct.
    # Row 1: [0, 0, 1, -1]. Correct.
    # Row 2: [0, 1, 0, 1]. Correct.
    # Row 3: [1, 0, 1, 0]. Correct.
    # Row 4: [-1, 1, 0, 0]. Correct.
    # Row 5: [0, 0, 0, 1]. Correct.
    # Row 6: [1, 0, 0, -1]. Correct.
    
    # Now, build the 4x4x4 decomposition using these 2x2x2 blocks.
    # A is 4x4. Split into 4 2x2 blocks: A00, A01, A10, A11.
    # B is 4x4. Split into 4 2x2 blocks: B00, B01, B10, B11.
    # C is 4x4. Split into 4 2x2 blocks: C00, C01, C10, C11.
    
    # The multiplication C = AB can be written in blocks:
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # We can apply Strassen to the 2x2x2 block multiplication.
    # But the blocks are 2x2 matrices.
    # So we are multiplying 2x2 matrices.
    # We use the 2x2x2 Strassen decomposition (rank 7) for each of the 7 "super-terms"?
    # No, we apply Strassen to the 4x4x4 problem directly by treating the 2x2 blocks as scalars?
    # No, the blocks are matrices.
    # We can use the 2x2x2 Strassen formula on the 2x2 blocks.
    # This requires 7 multiplications of 2x2 matrices.
    # Each 2x2 matrix multiplication has rank 7.
    # So total rank = 7 * 7 = 49.
    
    # Let's construct the 49 terms.
    # For each of the 7 Strassen steps (outer):
    #   We have linear combinations of the 2x2 blocks of A and B.
    #   Let A_outer be the 4x4 matrix formed by these combinations.
    #   Let B_outer be the 4x4 matrix formed by these combinations.
    #   We compute M_outer = A_outer B_outer.
    #   This is a 4x4 matrix product.
    #   We decompose this product using the 2x2x2 Strassen terms (inner).
    #   Wait, A_outer and B_outer are 4x4 matrices composed of 2x2 blocks.
    #   The product M_outer is a 4x4 matrix.
    #   We can decompose the multiplication of two 4x4 matrices into 7 multiplications of 2x2 matrices?
    #   No, the 2x2x2 Strassen formula applies to 2x2 matrices.
    #   Here we are multiplying 4x4 matrices.
    #   However, we can view the 4x4 multiplication as a 2x2 multiplication of 2x2 blocks.
    #   So we apply the 2x2x2 Strassen formula to the 2x2 blocks of A_outer and B_outer.
    #   This gives 7 terms.
    #   Each term involves a product of 2x2 matrices (the blocks).
    #   We then expand each 2x2 matrix product using the 2x2x2 Strassen formula (rank 7).
    #   So for each of the 7 outer terms, we add 7 inner terms.
    #   Total 49 terms.
    
    # Let's implement this.
    
    def get_4x4x4_terms():
        terms = []
        # Outer Strassen forms for 4x4 (using 2x2 blocks)
        # A_blocks: A00, A01, A10, A11
        # B_blocks: B00, B01, B10, B11
        
        # Outer Term 1: (A00+A11)(B00+B11)
        # Outer Term 2: (A10+A11)B00
        # Outer Term 3: A00(B01-B11)
        # Outer Term 4: A11(B10-B00)
        # Outer Term 5: (A00+A01)B11
        # Outer Term 6: (A10-A00)(B00+B01)
        # Outer Term 7: (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each outer term r:
        #   A_r is a 4x4 matrix.
        #   B_r is a 4x4 matrix.
        #   We need to add the decomposition of A_r B_r to C with the appropriate coefficients.
        
        # Let's define the outer coefficients.
        # A_outer_coefs: list of 7 dicts {(block_idx): coef}
        # B_outer_coefs: list of 7 dicts {(block_idx): coef}
        # C_outer_coefs: list of 7 dicts {(block_idx): coef}
        
        # Block indices: 0:00, 1:01, 2:10, 3:11
        
        A_outer = [
            {0: 1, 3: 1}, # M1
            {2: 1, 3: 1}, # M2
            {0: 1},       # M3
            {3: 1},       # M4
            {0: 1, 1: 1}, # M5
            {2: 1, 0: -1},# M6
            {1: 1, 3: -1} # M7
        ]
        
        B_outer = [
            {0: 1, 3: 1}, # M1
            {0: 1},       # M2
            {1: 1, 3: -1},# M3
            {2: 1, 0: -1},# M4
            {3: 1},       # M5
            {0: 1, 1: 1}, # M6
            {2: 1, 3: 1}  # M7
        ]
        
        C_outer = [
            {0: 1, 3: -1}, # M1
            {2: 1, 3: -1}, # M2
            {1: 1, 3: 1},  # M3
            {0: 1, 2: 1},  # M4
            {0: -1, 1: 1}, # M5
            {3: 1},        # M6
            {0: 1, 3: -1}  # M7
        ]
        
        # Now, for each outer term r:
        #   We have A_r and B_r.
        #   A_r is a 4x4 matrix. Its 2x2 blocks are linear combinations of the original 2x2 blocks of A.
        #   Let A_r_blocks[b] = sum_{b'} coef[b'] * A_blocks[b']
        #   Similarly for B_r.
        
        # We want to compute A_r B_r.
        # This is a 4x4 matrix product.
        # We can decompose this product using the 2x2x2 Strassen formula applied to the 2x2 blocks of A_r and B_r.
        # Let A_r have blocks A_r00, A_r01, A_r10, A_r11.
        # Let B_r have blocks B_r00, B_r01, B_r10, B_r11.
        
        # The 2x2x2 Strassen formula for A_r B_r gives 7 terms.
        # Each term involves a product of 2x2 matrices.
        # e.g., Term 1: (A_r00 + A_r11)(B_r00 + B_r11)
        # This is a product of two 2x2 matrices.
        # We decompose this product using the 2x2x2 Strassen formula (inner).
        # The inner formula gives 7 rank-1 terms.
        # Each rank-1 term is a scalar product of 2x2 matrices?
        # No, the inner formula decomposes the multiplication of two 2x2 matrices into 7 scalar multiplications.
        # So the product (A_r00 + A_r11)(B_r00 + B_r11) is a sum of 7 terms, each involving a scalar multiplication of entries of the 2x2 matrices.
        
        # This is getting deep.
        # Let's just build the U, V, W matrices directly by iterating over the 49 terms.
        
        # We will store the terms as (u_vec, v_vec, w_vec) where u_vec is length 16, v_vec length 16, w_vec length 16.
        # For 4x4x4.
        
        # For each outer term r (0..6):
        #   For each inner term s (0..6):
        #       Construct u, v, w.
        
        for r in range(7):
            # Construct A_r and B_r blocks
            # A_r_blocks is a 2x2 array of 2x2 matrices (represented as dicts of original block indices)
            # Actually, we can just track the coefficients of the original A entries.
            
            # Let's expand the inner Strassen terms for the product of A_r and B_r.
            # The inner Strassen terms are defined on the 2x2 blocks of A_r and B_r.
            # Let A_r = [Ar00, Ar01; Ar10, Ar11]
            # Let B_r = [Br00, Br01; Br10, Br11]
            
            # Inner Term 1: (Ar00+Ar11)(Br00+Br11)
            # Inner Term 2: (Ar10+Ar11)Br00
            # ...
            
            # We need to map Ar00 to original A blocks.
            # Ar00 = sum_{b'} A_outer[r][b'] * A_blocks[b']
            
            # Let's pre-compute the mapping of Ar and Br blocks to original A and B blocks.
            
            Ar_blocks = {} # key: (row, col) in 2x2 block grid, value: dict {orig_block: coef}
            Br_blocks = {}
            
            for br in range(2):
                for bc in range(2):
                    Ar_blocks[(br, bc)] = A_outer[r].get((br, bc), 0) # This is wrong. A_outer[r] is for the whole 4x4.
                    
            # A_outer[r] gives coefficients for the 4 blocks of A.
            # A_r is a 4x4 matrix.
            # The blocks of A_r are the same as the blocks of A, but scaled?
            # No. A_r is formed by linear combinations of the blocks of A.
            # e.g., M1 = (A00+A11)(B00+B11).
            # A_r = A00 + A11.
            # So A_r has blocks:
            # Ar00 = A00, Ar01 = 0, Ar10 = 0, Ar11 = A11.
            
            # So for each outer term r, we define Ar_blocks and Br_blocks.
            
            Ar_blocks = {}
            Br_blocks = {}
            
            for br in range(2):
                for bc in range(2):
                    # The block (br, bc) of A_r is the block (br, bc) of A, scaled by the coefficient in A_outer[r]?
                    # No, A_outer[r] is a dict {block_idx: coef}.
                    # block_idx 0 is (0,0), 1 is (0,1), 2 is (1,0), 3 is (1,1).
                    # So Ar_blocks[(br, bc)] = A_outer[r].get(br*2+bc, 0) * A_blocks[(br, bc)]
                    # But A_blocks are the identity.
                    # So Ar_blocks[(br, bc)] is just the coefficient.
                    # Wait, A_r is a 4x4 matrix.
                    # The block (br, bc) of A_r is the matrix A_blocks[(br, bc)] * coef.
                    # So the "content" of the block is the original A block scaled.
                    
                    Ar_blocks[(br, bc)] = A_outer[r].get(br*2+bc, 0)
                    Br_blocks[(br, bc)] = B_outer[r].get(br*2+bc, 0)
            
            # Now apply inner Strassen to Ar_blocks and Br_blocks.
            # Inner terms 1..7.
            
            # Inner Term 1: (Ar00+Ar11)(Br00+Br11)
            # A-part: Ar00 + Ar11
            # This is a 2x2 matrix.
            # Ar00 is A00 * coef_Ar00.
            # Ar11 is A11 * coef_Ar11.
            # So the A-part is (coef_Ar00 * A00) + (coef_Ar11 * A11).
            
            # We need to add a rank-1 term for the product of this A-part and the B-part.
            # The product is a 2x2 matrix.
            # We decompose this 2x2x2 product using the 2x2x2 Strassen formula.
            # This adds 7 terms.
            
            # For each inner term s (0..6):
            #   We have A_inner_s, B_inner_s, C_inner_s.
            #   A_inner_s is a linear combination of the entries of the A-part matrix.
            #   The A-part matrix is a linear combination of A00 and A11 (for example).
            #   So A_inner_s is a linear combination of the entries of A00 and A11.
            
            # This is a nested loop.
            # Outer r (7)
            #   Inner s (7)
            #     Inner-inner t (7)
            # Total 343 terms?
            # No, the rank of 4x4x4 is 49.
            # The rank of 2x2x2 is 7.
            # The rank of 4x4x4 is 7 * 7 = 49.
            # So we should have 49 terms.
            # My nested loop logic is wrong.
            
            # The 49 terms correspond to the 7 outer terms, each decomposed into 7 inner terms.
            # But the inner terms are rank-1 terms of the 2x2x2 multiplication.
            # So for each outer term r, we add 7 rank-1 terms.
            # Total 49.
            
            # Let's implement this.
            
            for s in range(7):
                # Inner term s of the 2x2x2 multiplication of Ar and Br.
                # A-part: U2x2[s] dot Ar_flattened
                # B-part: V2x2[s] dot Br_flattened
                # C-part: W2x2[s]
                
                # Ar_flattened is a vector of 4 elements, each being a 2x2 matrix (block).
                # Ar_flattened[0] = Ar00, Ar_flattened[1] = Ar01, etc.
                
                # U2x2[s] is a vector of 4 coefficients.
                # A-part = sum_{k=0}^3 U2x2[s][k] * Ar_flattened[k]
                
                # Ar_flattened[k] is a 2x2 matrix (block of A_r).
                # Ar_flattened[k] = coef * A_blocks[k]
                
                # So A-part is a 2x2 matrix.
                # A-part = sum_{k=0}^3 U2x2[s][k] * Ar_blocks[k] * A_blocks[k]
                
                # We need to expand this into the original A entries.
                
                # Let's build the u_vec (length 16) for this term.
                u_vec = [0.0] * 16
                v_vec = [0.0] * 16
                w_vec = [0.0] * 16
                
                # For each k in 0..3:
                #   coef = U2x2[s][k] * Ar_blocks[k]
                #   Add coef * A_blocks[k] to u_vec.
                
                for k in range(4):
                    coef = U2x2[s][k] * Ar_blocks.get((k//2, k%2), 0)
                    if coef != 0:
                        # Add coef * A_blocks[k] to u_vec
                        # A_blocks[k] is the 2x2 block of A at position k.
                        # k=0: (0,0), k=1: (0,1), k=2: (1,0), k=3: (1,1)
                        # Block (br, bc) occupies rows 2br..2br+1, cols 2bc..2bc+1.
                        br = k // 2
                        bc = k % 2
                        for dr in range(2):
                            for dc in range(2):
                                i = 2*br + dr
                                j = 2*bc + dc
                                u_vec[i*4 + j] += coef
                
                for k in range(4):
                    coef = V2x2[s][k] * Br_blocks.get((k//2, k%2), 0)
                    if coef != 0:
                        br = k // 2
                        bc = k % 2
                        for dr in range(2):
                            for dc in range(2):
                                i = 2*br + dr
                                j = 2*bc + dc
                                v_vec[i*4 + j] += coef
                
                # C-part
                # C_part = W2x2[s] dot C_blocks
                # C_blocks are the 2x2 blocks of C_r.
                # C_r is the result of A_r B_r.
                # The C-blocks are linear combinations of the original C blocks.
                # C_outer[r] gives the coefficients.
                
                # W2x2[s] is a vector of 4 coefficients for the 2x2 blocks of C_r.
                # C_r_flattened[k] = C_blocks[k] * C_outer[r][k]
                
                for k in range(4):
                    coef = W2x2[s][k] * C_outer[r].get(k, 0)
                    if coef != 0:
                        br = k // 2
                        bc = k % 2
                        for dr in range(2):
                            for dc in range(2):
                                i = 2*br + dr
                                j = 2*bc + dc
                                w_vec[i*4 + j] += coef
                
                terms.append((u_vec, v_vec, w_vec))
                
        return terms

    terms_4x4x4 = get_4x4x4_terms()
    
    # Now for 4x4x2.
    # Split B into B1 (4x2).
    # C is 4x2.
    # We can use the same logic, but B is 4x2.
    # We can pad B to 4x4 with zeros?
    # Or just implement 4x4x2 directly.
    # Rank 28.
    
    def get_4x4x2_terms():
        terms = []
        # A is 4x4. B is 4x2. C is 4x2.
        # Split A into 2x2 blocks.
        # Split B into 2x2 blocks? B is 4x2.
        # B00 (2x2), B10 (2x2).
        # C00 (2x2), C10 (2x2).
        
        # C00 = A00B00 + A01B10
        # C10 = A10B00 + A11B10
        
        # This is a 2x1 block multiplication.
        # We can use Strassen for 2x1?
        # Or just use schoolbook for the 2x2x2 products?
        # 4 products of 2x2x2.
        # Each 2x2x2 has rank 7.
        # Total rank 28.
        
        # Products:
        # 1. A00 B00 -> C00
        # 2. A01 B10 -> C00
        # 3. A10 B00 -> C10
        # 4. A11 B10 -> C10
        
        # For each product, add the 7 Strassen terms.
        
        products = [
            (0, 0, 0), # A00, B00, C00
            (1, 1, 0), # A01, B10, C00  (Note: B10 is the second block of B)
            (2, 0, 1), # A10, B00, C10
            (3, 1, 1)  # A11, B10, C10
        ]
        
        for a_blk, b_blk, c_blk in products:
            # a_blk: 0:00, 1:01, 2:10, 3:11
            # b_blk: 0:00 (top), 1:10 (bottom)
            # c_blk: 0:00 (top), 1:10 (bottom)
            
            for s in range(7):
                u_vec = [0.0] * 16
                v_vec = [0.0] * 12 # 4x2 = 8? No, B is 4x2, so 8 entries. Wait.
                # B is m x p = 4 x 2.
                # Flattened V is length m*p = 8.
                # My previous v_vec was length 16 for 4x4.
                # Here v_vec should be length 8.
                
                # Let's reset v_vec size.
                v_vec = [0.0] * 8
                w_vec = [0.0] * 8 # C is 4x2, so 8 entries.
                
                # A-part
                # A_blk is a 2x2 block.
                # U2x2[s] gives coefficients for the 4 entries of the 2x2 A block.
                # But we are multiplying the whole block.
                # So we use the 2x2x2 Strassen terms on the block.
                
                # For the product A_blk * B_blk:
                # A_blk is 2x2. B_blk is 2x2.
                # We use the 7 terms of 2x2x2.
                
                # A-part: U2x2[s] dot A_blk
                # A_blk is at position a_blk in A.
                # a_blk 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                
                # We need to map the local 2x2 indices to global 4x2 indices.
                
                # A-side:
                # Local (i, j) in 2x2 block -> Global (2*br+i, 2*bc+j)
                # a_blk 0: br=0, bc=0
                # a_blk 1: br=0, bc=1
                # a_blk 2: br=1, bc=0
                # a_blk 3: br=1, bc=1
                
                br_a = a_blk // 2
                bc_a = a_blk % 2
                
                for li in range(2):
                    for lj in range(2):
                        coef = U2x2[s][li*2 + lj]
                        if coef != 0:
                            gi = 2*br_a + li
                            gj = 2*bc_a + lj
                            u_vec[gi*4 + gj] += coef
                
                # B-side:
                # B_blk is at position b_blk in B.
                # b_blk 0: top 2x2 (rows 0-1, cols 0-1)
                # b_blk 1: bottom 2x2 (rows 2-3, cols 0-1)
                
                br_b = b_blk # 0 or 1
                bc_b = 0 # B is 4x2, so only 1 col block of width 2.
                
                for li in range(2):
                    for lj in range(2):
                        coef = V2x2[s][li*2 + lj]
                        if coef != 0:
                            gi = 2*br_b + li
                            gj = 2*bc_b + lj
                            v_vec[gi*2 + gj] += coef
                
                # C-side:
                # C_blk is at position c_blk in C.
                # c_blk 0: top 2x2
                # c_blk 1: bottom 2x2
                
                br_c = c_blk
                bc_c = 0
                
                for li in range(2):
                    for lj in range(2):
                        coef = W2x2[s][li*2 + lj]
                        if coef != 0:
                            gi = 2*br_c + li
                            gj = 2*bc_c + lj
                            w_vec[gi*2 + gj] += coef
                
                terms.append((u_vec, v_vec, w_vec))
                
        return terms

    terms_4x4x2 = get_4x4x2_terms()
    
    # Combine terms
    # 4x4x4 terms: u (16), v (16), w (16)
    # 4x4x2 terms: u (16), v (8), w (8)
    
    # We need to pad v and w for 4x4x2 to match the 4x4x4 dimensions?
    # No, the final V is (m*p, rank) = (24, rank).
    # The 4x4x4 part uses B1 (4x4), which is the first 16 entries of B (flattened).
    # The 4x4x2 part uses B2 (4x2), which is the last 8 entries of B.
    
    # So for 4x4x2 terms, v_vec is length 8.
    # We need to place it in the last 8 columns of V.
    
    # Similarly for W.
    # 4x4x4 uses C1 (4x4), first 16 entries.
    # 4x4x2 uses C2 (4x2), last 8 entries.
    
    total_rank = len(terms_4x4x4) + len(terms_4x4x2)
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(24)]
    W = [[0] * total_rank for _ in range(24)]
    
    col = 0
    for u, v, w in terms_4x4x4:
        for i in range(16):
            U[i][col] = u[i]
        for i in range(16):
            V[i][col] = v[i]
        for i in range(16):
            W[i][col] = w[i]
        col += 1
        
    for u, v, w in terms_4x4x2:
        for i in range(16):
            U[i][col] = u[i]
        for i in range(8):
            V[16+i][col] = v[i]
        for i in range(8):
            W[16+i][col] = w[i]
        col += 1
        
    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check rank
    # If any column is zero, rank is less.
    
    # Check correctness
    # C = W^T * (U^T A V) ?
    # C_{ki} = sum_r W_{kr} U_{ir} V_{jr} ?
    # No, C_{ki} = sum_r W_{kr} (sum_{ij} U_{ir} A_{ij} V_{jr})
    
    # Let's test with random A, B.
    A_test = np.random.rand(4, 4)
    B_test = np.random.rand(4, 6)
    C_test = A_test @ B_test
    
    # Compute C_decomp
    # M_r = U_r^T A V_r
    # C = sum_r W_r M_r
    
    # U_r is column r of U.
    # V_r is column r of V.
    # W_r is column r of W.
    
    C_decomp = np.zeros((4, 6))
    for r in range(total_rank):
        u_r = U_np[:, r]
        v_r = V_np[:, r]
        w_r = W_np[:, r]
        
        # u_r is length 16 (4x4)
        # v_r is length 24 (4x6)
        # w_r is length 24 (6x4)
        
        u_mat = u_r.reshape(4, 4)
        v_mat = v_r.reshape(4, 6)
        w_mat = w_r.reshape(6, 4)
        
        # M = u_mat @ A_test @ v_mat ?
        # No, the term is (u_r \cdot A) (v_r \cdot B) w_r ?
        # The decomposition is C = sum_r (U_r^T A) (V_r^T B) W_r ?
        # No, U_r is a vector.
        # The rank-1 term is U_r \otimes V_r \otimes W_r.
        # C_{ki} = sum_r W_{kr} (sum_{ij} U_{ir} A_{ij} V_{jr})
        
        # Let's compute the scalar M_r = sum_{ij} U_{ir} A_{ij} V_{jr}
        # U_{ir} is U[i, r]
        # V_{jr} is V[j, r]
        
        M_r = 0
        for i in range(4):
            for j in range(6):
                M_r += U_np[i*4+j, r] * A_test[i, j] * V_np[j*6+i, r] # Wait, V index is j*p+k?
                # V is (m*p, rank). m=4, p=6.
                # V[j*6+k, r]
                # A is (4, 4). A[i, j]
                # B is (4, 6). B[j, k]
                # So M_r = sum_{i, j, k} U[i, r] A[i, j] V[j, r] ?
                # No, V depends on j and k.
                # The term is sum_{i, j, k} U[i, r] A[i, j] V[j, r] B[j, k] ?
                # No, V is a factor for B.
                # The product is (U_r \cdot A) (V_r \cdot B) ?
                # No, U_r and V_r are vectors.
                # The rank-1 tensor is T_{ijk} = U_{i} V_{j} W_{k}.
                # C_{ik} = sum_j sum_r U_{ir} A_{ij} V_{jr} W_{kr}
                
                # Let's re-verify the indices.
                # U is (n*m, rank). Index i*m+j.
                # V is (m*p, rank). Index j*p+k.
                # W is (p*n, rank). Index k*n+i.
                
                # C_{ik} = sum_r W_{k*n+i, r} * (sum_{j} U_{i*m+j, r} A_{ij} V_{j*p+k, r})
                
                # My loop above was wrong.
        
        # Let's rewrite the verification.
        
    C_decomp = np.zeros((4, 6))
    for r in range(total_rank):
        for i in range(4):
            for k in range(6):
                w_val = W_np[k*4 + i, r]
                if w_val == 0: continue
                sum_val = 0
                for j in range(4):
                    u_val = U_np[i*4 + j, r]
                    if u_val == 0: continue
                    a_val = A_test[i, j]
                    if a_val == 0: continue
                    v_val = V_np[j*6 + k, r]
                    sum_val += u_val * a_val * v_val
                C_decomp[i, k] += w_val * sum_val
                
    if not np.allclose(C_test, C_decomp):
        # Fallback to schoolbook
        print("Strassen failed, using schoolbook")
        rank = 96
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(24)]
        W = [[0] * rank for _ in range(24)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j][col] = 1
                    V[j*6+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
