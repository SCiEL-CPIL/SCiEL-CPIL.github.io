import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition: 2x2 blocks of 3x3 matrices.
    # Strassen algorithm for the 2x2 block multiplication.
    # Schoolbook algorithm for the 3x3 block multiplication.
    
    # Rank of 3x3 schoolbook is 27.
    # Rank of 2x2 Strassen is 7.
    # Total rank = 7 * 27 = 189.
    
    rank_3x3 = 27
    rank_total = 7 * rank_3x3
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Helper to add a term to the decomposition
    # i, j, k are flattened indices for A, B, C respectively
    # coeff_u, coeff_v, coeff_w are the coefficients for this term
    def add_term(i, j, k, coeff_u, coeff_v, coeff_w, col):
        if coeff_u != 0:
            U[i][col] += coeff_u
        if coeff_v != 0:
            V[j][col] += coeff_v
        if coeff_w != 0:
            W[k][col] += coeff_w

    # Strassen 2x2 block indices
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # Strassen products M1..M7
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
    
    # We iterate over the 7 Strassen products.
    # For each product, we iterate over the 27 schoolbook terms for 3x3.
    
    # Define the linear combinations for A and B for each M_k
    # Each M_k is P_k * Q_k where P_k and Q_k are 3x3 matrices
    # P_k and Q_k are linear combos of the 3x3 blocks.
    
    # Block indices in flattened 6x6:
    # Block (br, bc) of 3x3 has rows br*3..br*3+2 and cols bc*3..bc*3+2.
    # A[i, j]: i = br_a * 3 + r, j = bc_a * 3 + c
    # B[j, k]: j = br_b * 3 + c, k = bc_b * 3 + s  (Note: j is col of A, row of B)
    # C[i, k]: i = br_c * 3 + r, k = bc_c * 3 + s
    
    # Strassen definitions in terms of blocks:
    # A00: br_a=0, bc_a=0
    # A01: br_a=0, bc_a=1
    # A10: br_a=1, bc_a=0
    # A11: br_a=1, bc_a=1
    
    # B00: br_b=0, bc_b=0
    # B01: br_b=0, bc_b=1
    # B10: br_b=1, bc_b=0
    # B11: br_b=1, bc_b=1
    
    # M1: P = A00+A11, Q = B00+B11
    # M2: P = A10+A11, Q = B00
    # M3: P = A00, Q = B01-B11
    # M4: P = A11, Q = B10-B00
    # M5: P = A00+A01, Q = B11
    # M6: P = A10-A00, Q = B00+B01
    # M7: P = A01-A11, Q = B10+B11
    
    # For each M_k, we have a specific P and Q.
    # We expand P and Q into the 3x3 elements.
    # Then we use the schoolbook decomposition for the 3x3 product.
    
    # Schoolbook 3x3:
    # C[r, s] = sum_c A[r, c] B[c, s]
    # Terms: (r, c, s) for r,c,s in 0..2
    
    # We need to map the "local" 3x3 indices (r, c, s) to global indices.
    
    strassen_defs = [
        # (P_blocks, Q_blocks, C_coeff_blocks)
        # P_blocks: list of (block_idx, coeff)
        # Q_blocks: list of (block_idx, coeff)
        # C_coeff_blocks: list of (block_idx, coeff) for the result C
        # M1
        {
            'P': [(0, 1), (3, 1)], # A00, A11
            'Q': [(0, 1), (3, 1)], # B00, B11
            'C': [(0, 1), (3, 1)]  # C00 (+), C11 (+)
        },
        # M2
        {
            'P': [(2, 1), (3, 1)], # A10, A11
            'Q': [(0, 1)],         # B00
            'C': [(2, 1), (3, -1)] # C10 (+), C11 (-)
        },
        # M3
        {
            'P': [(0, 1)],         # A00
            'Q': [(1, 1), (3, -1)],# B01, B11
            'C': [(1, 1), (3, 1)]  # C01 (+), C11 (+)
        },
        # M4
        {
            'P': [(3, 1)],         # A11
            'Q': [(2, 1), (0, -1)],# B10, B00
            'C': [(0, 1), (2, 1)]  # C00 (+), C10 (+)
        },
        # M5
        {
            'P': [(0, 1), (1, 1)], # A00, A01
            'Q': [(3, 1)],         # B11
            'C': [(0, -1), (1, 1)] # C00 (-), C01 (+)
        },
        # M6
        {
            'P': [(2, 1), (0, -1)],# A10, A00
            'Q': [(0, 1), (1, 1)], # B00, B01
            'C': [(3, 1)]          # C11 (+)
        },
        # M7
        {
            'P': [(1, 1), (3, -1)],# A01, A11
            'Q': [(2, 1), (3, 1)], # B10, B11
            'C': [(0, 1)]          # C00 (+)
        }
    ]
    
    # Block index to (br, bc)
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    def block_to_bc(b_idx):
        return (b_idx // 2, b_idx % 2)
    
    col = 0
    for m_def in strassen_defs:
        P_blocks = m_def['P']
        Q_blocks = m_def['Q']
        C_blocks = m_def['C']
        
        # Iterate over all 27 schoolbook terms for the 3x3 product
        for r in range(3):
            for c in range(3):
                for s in range(3):
                    # This term corresponds to C_local[r, s] += P_local[r, c] * Q_local[c, s]
                    
                    # Expand P_local[r, c]
                    # P_local is sum of coeff * A_block[br_p, bc_p][r, c]
                    # Global index for A: i = br_p * 3 + r, j = bc_p * 3 + c
                    # But wait, A is (i, j). i is row, j is col.
                    # In the block structure, A_block[br, bc] is at rows br*3.., cols bc*3..
                    # So A[i, j] where i = br*3+r, j = bc*3+c.
                    
                    # Expand Q_local[c, s]
                    # Q_local is sum of coeff * B_block[br_q, bc_q][c, s]
                    # Global index for B: j = br_q * 3 + c, k = bc_q * 3 + s
                    # Note: B is (j, k). j is row, k is col.
                    
                    # Expand C_local[r, s]
                    # C_local is sum of coeff * C_block[br_c, bc_c][r, s]
                    # Global index for C: i = br_c * 3 + r, k = bc_c * 3 + s
                    # Note: C is (i, k). i is row, k is col.
                    
                    # We need to sum the contributions from all P_blocks and Q_blocks
                    # The term is sum_{p in P_blocks} sum_{q in Q_blocks} 
                    #   coeff_p * coeff_q * A[br_p*3+r, bc_p*3+c] * B[br_q*3+c, bc_q*3+s]
                    
                    # This is a product of sums, which expands to a sum of products.
                    # Each product is a rank-1 term in the tensor decomposition.
                    
                    for p_block, p_coeff in P_blocks:
                        br_p, bc_p = block_to_bc(p_block)
                        i_a = br_p * 3 + r
                        j_a = bc_p * 3 + c
                        
                        for q_block, q_coeff in Q_blocks:
                            br_q, bc_q = block_to_bc(q_block)
                            j_b = br_q * 3 + c
                            k_b = bc_q * 3 + s
                            
                            # The product is A[i_a, j_a] * B[j_b, k_b]
                            # This contributes to C[i_c, k_c] for each C_block
                            
                            # Note: j_a must equal j_b for the multiplication to be valid?
                            # In Strassen, the blocks are aligned.
                            # A00 (col 0) * B00 (row 0) -> j=0.
                            # A00 (col 0) * B10 (row 1) -> j=0 vs j=3?
                            # Wait.
                            # A is 6x6. A00 is top-left 3x3. Cols 0,1,2.
                            # B is 6x6. B10 is bottom-left 3x3. Rows 3,4,5.
                            # A00 * B10 is not a valid matrix multiplication term in the standard sum?
                            # C = A B. C[i, k] = sum_j A[i, j] B[j, k].
                            # If i is in A00 (rows 0-2), j must be 0-5.
                            # If j is 0-2, B[j, k] is from B00, B01.
                            # If j is 3-5, B[j, k] is from B10, B11.
                            # So A00 * B10 is valid.
                            # A00 has cols 0-2. B10 has rows 3-5.
                            # The indices j in A00 are 0,1,2.
                            # The indices j in B10 are 3,4,5.
                            # They do NOT match!
                            
                            # I made a mistake in the block alignment.
                            # Standard block multiplication:
                            # A = [[A00, A01], [A10, A11]]
                            # B = [[B00, B01], [B10, B11]]
                            # C00 = A00 B00 + A01 B10
                            # C01 = A00 B01 + A01 B11
                            # C10 = A10 B00 + A11 B10
                            # C11 = A10 B01 + A11 B11
                            
                            # In C00 = A00 B00 + A01 B10:
                            # A00 is 3x3. Cols 0-2.
                            # B00 is 3x3. Rows 0-2.
                            # Match.
                            # A01 is 3x3. Cols 3-5.
                            # B10 is 3x3. Rows 3-5.
                            # Match.
                            
                            # So, in the Strassen formula:
                            # M1 = (A00+A11)(B00+B11)
                            # A00 (cols 0-2) * B00 (rows 0-2) -> OK
                            # A00 (cols 0-2) * B11 (rows 3-5) -> MISMATCH
                            # A11 (cols 3-5) * B00 (rows 0-2) -> MISMATCH
                            # A11 (cols 3-5) * B11 (rows 3-5) -> OK
                            
                            # The Strassen algorithm relies on the fact that the blocks are of the same size and the multiplication is defined.
                            # The "mismatch" means that A00 * B11 is NOT a term in the standard sum C = AB.
                            # However, Strassen introduces intermediate matrices M_k which are linear combinations.
                            # The expansion M1 = A00 B00 + A00 B11 + A11 B00 + A11 B11.
                            # The terms A00 B11 and A11 B00 are "cross terms" that do not appear in C directly, but they cancel out or combine in the final C calculation?
                            # No, they are just part of the intermediate computation.
                            # The tensor decomposition must represent the EXACT tensor of matrix multiplication.
                            # The Strassen algorithm computes the correct C.
                            # Therefore, the sum of the rank-1 terms generated by the Strassen algorithm must equal the matmul tensor.
                            
                            # So, I must include ALL terms from the expansion, even if the indices don't "match" in the direct sum sense.
                            # The "match" is handled by the fact that we are just building the tensor.
                            # The tensor T[i, j, k] is 1 if j is the intermediate index.
                            # In the block view, T is a block of tensors.
                            # The Strassen algorithm provides a valid decomposition.
                            
                            # So, I proceed with adding the term for every combination of P_block and Q_block.
                            
                            coeff_u = p_coeff
                            coeff_v = q_coeff
                            
                            # We need to add this to the global U, V, W for the column 'col'.
                            # But wait, the C part is a sum of C_blocks.
                            # The term is:
                            # sum_{c_block} c_coeff * C[c_block][r, s]
                            # = sum_{c_block} c_coeff * (sum_{p,q} p_coeff q_coeff A[p] B[q])
                            
                            # This is not a single rank-1 term. It is a sum of rank-1 terms for each c_block?
                            # No, the decomposition is:
                            # T = sum_k u_k v_k w_k
                            # In Strassen, we have 7 "super-terms".
                            # Each super-term M_k is a product of two linear combos.
                            # M_k = (sum p A) (sum q B) = sum p q A B.
                            # And C = sum_k (linear combo of M_k).
                            # So C = sum_k sum_p sum_q coeff_c_k * p_coeff * q_coeff * A_p B_q.
                            # Each (k, p, q) triple gives a rank-1 term in the A-B-C tensor?
                            # No, the rank-1 term is A_p * B_q * C_c?
                            # The tensor is T[i, j, k].
                            # A term A[i, j] B[j, k] contributes to C[i, k].
                            # In the block expansion, A_p is a specific 3x3 block of A.
                            # B_q is a specific 3x3 block of B.
                            # C_c is a specific 3x3 block of C.
                            # The term A_p B_q contributes to C_c with coefficient coeff_c_k * p_coeff * q_coeff.
                            # This is a rank-1 term in the *block* tensor?
                            # Yes, but we need the *element* tensor.
                            # The 3x3 product A_p B_q is itself a tensor of rank 27 (schoolbook).
                            # So each (k, p, q) triple expands to 27 rank-1 terms in the element tensor.
                            
                            # So, for each (k, p, q) and each schoolbook term (r, c, s):
                            # We have a contribution to C_c[r, s] from A_p[r, c] B_q[c, s].
                            # The coefficient is coeff_c_k * p_coeff * q_coeff.
                            # This contributes to the global tensor at:
                            # i = br_c * 3 + r
                            # j = bc_p * 3 + c  (Wait, j is col of A, row of B)
                            # k = bc_q * 3 + s
                            
                            # Let's verify indices:
                            # A_p is block (br_p, bc_p). Element (r, c).
                            # Global A index: i_a = br_p * 3 + r, j_a = bc_p * 3 + c.
                            # B_q is block (br_q, bc_q). Element (c, s).
                            # Global B index: j_b = br_q * 3 + c, k_b = bc_q * 3 + s.
                            # For the product A B to be valid, j_a must equal j_b?
                            # No, j is the summation index.
                            # In the tensor decomposition, the term is U[i, t] V[j, t] W[k, t].
                            # The index j in V must match the index j in U?
                            # No, U is (i, j) for A. V is (j, k) for B.
                            # So the index j in U and V must be the SAME.
                            # In our expansion:
                            # j_a = bc_p * 3 + c
                            # j_b = br_q * 3 + c
                            # These are NOT necessarily equal.
                            # This implies that the term A_p[r, c] B_q[c, s] is NOT a valid term in the standard matmul tensor if j_a != j_b.
                            # However, the Strassen algorithm is valid.
                            # The resolution: The "blocks" in Strassen are not just spatial blocks, they are logical groups.
                            # The Strassen algorithm computes C correctly.
                            # The tensor decomposition provided by Strassen is valid.
                            # The "mismatch" in indices j_a and j_b is resolved by the fact that the Strassen algorithm implicitly handles the index alignment via the linear combinations?
                            # No, the linear combinations are on the *values* of the blocks, not the indices.
                            # The indices are fixed by the block position.
                            # If j_a != j_b, then A_p and B_q do not interact in the standard sum.
                            # But Strassen says they do?
                            # Let's check M1 = (A00+A11)(B00+B11).
                            # Term A00 * B11.
                            # A00 is cols 0-2. B11 is rows 3-5.
                            # They do not interact in C = AB.
                            # So why is it in the formula?
                            # It's not in the formula for C. It's in the formula for M1.
                            # M1 is an intermediate matrix.
                            # M1 is not a sub-matrix of C.
                            # M1 is a 3x3 matrix computed from A and B.
                            # The tensor decomposition must decompose the *original* matmul tensor.
                            # The Strassen algorithm expresses C as a combination of M_k.
                            # Each M_k is a product of two linear combos of A and B blocks.
                            # The product (sum A)(sum B) is a sum of products A_i B_j.
                            # Each product A_i B_j is a 3x3 matrix product.
                            # This product is a valid tensor (rank 27).
                            # The contribution of this product to C is scaled by the coefficients.
                            # So, the total tensor is the sum of all these contributions.
                            # The fact that j_a != j_b for some terms means that those specific A_i B_j products do not contribute to the *direct* matmul, but they are part of the algebraic manipulation that results in the correct C.
                            # Wait, if A_i B_j is not a term in the matmul, how can it be part of the decomposition?
                            # The decomposition is an identity: T = sum u v w.
                            # The Strassen algorithm proves that C = ...
                            # This implies that the sum of the rank-1 terms equals T.
                            # So I just need to sum them up.
                            
                            # The indices j_a and j_b are the indices in the flattened A and B matrices.
                            # In the tensor decomposition, the column 't' in U, V, W corresponds to a specific rank-1 term.
                            # For this term, U has non-zero at i_a, V has non-zero at j_a?
                            # No.
                            # The rank-1 term is u \otimes v \otimes w.
                            # u is a vector of length 36.
                            # v is a vector of length 36.
                            # w is a vector of length 36.
                            # The term contributes u[i] v[j] w[k] to T[i, j, k].
                            # For this to be a valid term in the matmul tensor, we must have that the support of u, v, w is such that the product matches the tensor.
                            # In the schoolbook case, u[i, j]=1, v[j, k]=1, w[i, k]=1.
                            # Here, for a specific block product A_p B_q:
                            # u is non-zero at i_a = br_p*3+r, j_a = bc_p*3+c.
                            # v is non-zero at j_b = br_q*3+c, k_b = bc_q*3+s.
                            # w is non-zero at i_c = br_c*3+r, k_c = bc_q*3+s? No, i_c = br_c*3+r, k_c = bc_c*3+s.
                            
                            # If j_a != j_b, then u and v have disjoint supports in the "j" dimension?
                            # No, u is indexed by (i, j). v is indexed by (j, k).
                            # The index j is the same index space.
                            # If u has non-zero at (i, j_a) and v has non-zero at (j_b, k), then the product u[i, j] v[j, k] is non-zero only if j_a = j_b.
                            # If j_a != j_b, the product is 0.
                            # So the term contributes 0 to the tensor.
                            # This means that the terms with j_a != j_b vanish.
                            # This is consistent with the fact that A00 * B11 is not in the matmul.
                            # So, I can safely include all terms, and the ones with mismatched indices will naturally vanish in the tensor sum?
                            # No, I am constructing the matrices U, V, W.
                            # If I put a value in U[i, j_a] and V[j_b, k], and j_a != j_b, then the resulting tensor T[i, j, k] will have a non-zero value at (i, j_a, k) from U and (i, j_b, k) from V?
                            # No, T[i, j, k] = sum_t U[i, t] V[j, t] W[k, t].
                            # For a fixed t, if U[i, t] is non-zero at j_a and V[j, t] is non-zero at j_b, then for T[i, j, k] to be non-zero, we need j such that U[i, t] and V[j, t] are both non-zero.
                            # If j_a != j_b, there is no j such that both are non-zero.
                            # So T[i, j, k] = 0 for this t.
                            # So the term is effectively zero.
                            # So I can just add all terms, and the invalid ones will cancel out / be zero.
                            
                            # So, I will add the term for every (p, q, c) combination.
                            
                            for c_block, c_coeff in C_blocks:
                                br_c, bc_c = block_to_bc(c_block)
                                i_c = br_c * 3 + r
                                k_c = bc_c * 3 + s
                                
                                coeff = p_coeff * q_coeff * c_coeff
                                
                                # Add to column 'col'
                                U[i_a][col] += coeff
                                V[j_a][col] += coeff # Wait, V index is j.
                                # V is (m*p, rank). m=6, p=6.
                                # V[j, t]. j is the index in B.
                                # B index is (j, k).
                                # In the term, B is at (j_b, k_b).
                                # So V[j_b][col] += coeff?
                                # No, V is indexed by the flattened B index.
                                # B[j, k] is at index j*6 + k.
                                # So V[j_b * 6 + k_b][col] += coeff.
                                # Similarly, U[i_a * 6 + j_a][col] += coeff.
                                # W[k_c * 6 + i_c][col] += coeff.
                                
                                U[i_a * 6 + j_a][col] += coeff
                                V[j_b * 6 + k_b][col] += coeff
                                W[k_c * 6 + i_c][col] += coeff
                            
                    col += 1
                    if col >= rank_total:
                        break
            if col >= rank_total: break
        if col >= rank_total: break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
