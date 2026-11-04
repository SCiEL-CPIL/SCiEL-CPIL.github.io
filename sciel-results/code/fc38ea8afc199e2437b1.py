import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140
    
    # Rank-7 Strassen terms for 2x2x2
    # u, v, w are lists of 4 floats
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Function to expand a 2x2x2 term into a 4x4x4 term
    # block_idx: 0 (00), 1 (01), 2 (10), 3 (11)
    # Returns a list of (u, v, w) for the 4x4x4 case
    # u, v, w are lists of 16 floats
    def expand_term(u2, v2, w2, a_block, b_block, c_block):
        u = [0] * 16
        v = [0] * 16
        w = [0] * 16
        
        # Map local indices 0,1,2,3 to global indices for the block
        # Block 0: rows 0-1, cols 0-1 -> 0,1,4,5
        # Block 1: rows 0-1, cols 2-3 -> 2,3,6,7
        # Block 2: rows 2-3, cols 0-1 -> 8,9,12,13
        # Block 3: rows 2-3, cols 2-3 -> 10,11,14,15
        
        def get_global_indices(block):
            if block == 0: return [0, 1, 4, 5]
            if block == 1: return [2, 3, 6, 7]
            if block == 2: return [8, 9, 12, 13]
            if block == 3: return [10, 11, 14, 15]
            
        u_idx = get_global_indices(a_block)
        v_idx = get_global_indices(b_block)
        w_idx = get_global_indices(c_block)
        
        for i in range(4):
            u[u_idx[i]] = u2[i]
            v[v_idx[i]] = v2[i]
            w[w_idx[i]] = w2[i]
            
        return (u, v, w)

    # Generate 49 terms for 4x4x4
    terms_4x4 = []
    for (u2, v2, w2) in strassen_2x2:
        # We need to determine which 2x2 blocks of A, B, C are involved
        # The Strassen formulas for 4x4x4 are the same as 2x2x2 but with matrix blocks
        # M1 = (A00+A11)(B00+B11)
        # This means we add the terms for A00*B00, A00*B11, A11*B00, A11*B11?
        # No, Strassen is linear.
        # M1 = (A00+A11) * (B00+B11)
        # This product is a 2x2 matrix multiplication.
        # We decompose this 2x2 multiplication into 7 terms.
        # For each of these 7 terms, the A vector is (u2_A00 + u2_A11) placed in the A00 and A11 blocks?
        # No.
        # Let's look at the structure.
        # The top-level Strassen term k has u2, v2, w2.
        # u2 is a vector of 4 coefficients for the 4 corners of the 2x2 A matrix.
        # These corners correspond to the 4 blocks of the 4x4 A matrix.
        # So if u2[0] = 1, it means the top-left block A00 is included with coeff 1.
        # If u2[3] = 1, it means the bottom-right block A11 is included with coeff 1.
        
        # So for a given top-level term (u2, v2, w2):
        # We need to generate the 7 sub-terms for the 2x2 multiplication defined by the blocks selected by u2, v2, w2?
        # No, that's not right.
        # The top-level term is a single multiplication of two 2x2 matrices (blocks).
        # Let A' be the 2x2 matrix of blocks. A'_{ij} = A_{block(i,j)}.
        # Let B' be the 2x2 matrix of blocks.
        # The product M = A' B' is a 2x2 matrix of blocks.
        # We need to decompose this 2x2 matrix multiplication.
        # The rank-7 decomposition of 2x2x2 gives 7 terms.
        # Each term t in the rank-7 decomposition has vectors u_t, v_t, w_t of length 4.
        # u_t[i] corresponds to the coefficient of A'_{ij}? No, u_t is a vector of length 4 representing the linear combination of the 4 elements of the 2x2 matrix A'.
        # So u_t[0] is the coeff for A'_{00} (which is block A00).
        # u_t[1] is the coeff for A'_{01} (which is block A01).
        # u_t[2] is the coeff for A'_{10} (which is block A10).
        # u_t[3] is the coeff for A'_{11} (which is block A11).
        
        # So for each top-level term (u2, v2, w2):
        # u2 selects the blocks of A.
        # v2 selects the blocks of B.
        # w2 selects the blocks of C.
        
        # But wait, the top-level term is just ONE multiplication of two 2x2 matrices.
        # The Strassen algorithm for 4x4 uses 7 such multiplications.
        # Each multiplication is of two 2x2 matrices.
        # So we need to decompose each of these 7 multiplications.
        
        # For the k-th top-level term, let A'_k be the 2x2 matrix of blocks (linear comb of A00..A11).
        # Let B'_k be the 2x2 matrix of blocks.
        # We need to multiply A'_k * B'_k.
        # We use the rank-7 decomposition for this 2x2x2 multiplication.
        # The rank-7 decomposition produces 7 terms.
        # For each sub-term s, we have vectors u_s, v_s, w_s of length 4.
        # u_s[i] is the coefficient for the i-th block of A'_k.
        # But A'_k itself is a linear combination of the original blocks A00..A11.
        # This creates a nested structure.
        
        # Actually, it's simpler:
        # The 4x4x4 multiplication tensor is the Kronecker product of the 2x2x2 tensor with itself?
        # No, it's a recursive application.
        # Let T be the 2x2x2 tensor.
        # The 4x4x4 tensor is obtained by substituting each scalar in T with a 2x2x2 tensor?
        # Yes.
        # So we can generate the 49 terms by iterating over the 7 terms of the outer Strassen and the 7 terms of the inner Strassen.
        
        # Outer term k: (u2_k, v2_k, w2_k).
        # u2_k is a vector of 4 coeffs.
        # This means A'_k = \sum_{i=0}^3 u2_k[i] A_{block_i}.
        # Similarly for B'_k and C'_k.
        
        # Now we multiply A'_k * B'_k.
        # We use the 7 inner terms (u3_s, v3_s, w3_s).
        # The product is \sum_s (A'_k \cdot u3_s) * (B'_k \cdot v3_s) \to (C'_k \cdot w3_s).
        # Here \cdot means the linear combination.
        # A'_k \cdot u3_s = \sum_{i} u2_k[i] u3_s[i] A_{block_i}.
        # So the coefficient for A_{block_i} in the final u vector is u2_k[i] * u3_s[i].
        
        # So for each outer term k and inner term s:
        # u_final is a vector of 16.
        # For each block i (0..3):
        #   The 4 entries in block i of u_final are u2_k[i] * u3_s[i] * [1,0,0,0]? No.
        #   u3_s[i] is a scalar. It multiplies the entire block A_{block_i}.
        #   But A_{block_i} is a 2x2 matrix.
        #   The inner Strassen term s has u3_s which is a vector of 4 scalars for the 4 elements of the 2x2 matrix A'.
        #   Wait, A' is a 2x2 matrix of scalars? No, A' is a 2x2 matrix of 2x2 blocks.
        #   So A'_{ij} is a 2x2 block.
        #   The inner Strassen acts on the "block" level?
        #   No, the inner Strassen acts on the 2x2 matrix of blocks.
        #   So the "elements" of the 2x2 matrix in the inner Strassen are the blocks themselves.
        #   So u3_s[0] is a coefficient for the top-left block of A'.
        #   But the top-left block of A' is A_{block_0}.
        #   So the contribution to the global u vector is u2_k[0] * u3_s[0] * (the identity vector for block 0)?
        #   No, the inner Strassen decomposes the multiplication of two 2x2 matrices of blocks.
        #   The "entries" of these matrices are blocks.
        #   So the rank-1 term in the inner decomposition is a product of linear combinations of blocks.
        #   So the u vector for the global term will have non-zero entries in the blocks selected by u3_s.
        #   And the coefficients in those blocks will be u2_k[i] * u3_s[i].
        #   But wait, the blocks are 2x2 matrices. The u vector is flat.
        #   So if u3_s[0] = 1, it means the entire block A00 is included with coeff 1.
        #   So in the global u vector, the 4 entries corresponding to A00 are all 1?
        #   No, that would be a rank-1 term where u is constant on the block.
        #   This is correct for the "block" multiplication.
        #   But we are decomposing the multiplication of the blocks.
        #   The blocks are 2x2 matrices.
        #   The multiplication of two 2x2 matrices of blocks is equivalent to a 4x4 matrix multiplication.
        #   So the inner Strassen is just the 2x2 Strassen applied to the block indices.
        #   So yes, the u vector will have the same value for all 4 elements of a block if that block is selected.
        
        # Let's verify.
        # If we multiply two 2x2 matrices of blocks, say A' and B'.
        # C' = A' B'.
        # C'_{00} = A'_{00} B'_{00} + A'_{01} B'_{10}.
        # This is a sum of products of blocks.
        # Each product of blocks (e.g., A'_{00} B'_{00}) is a 2x2 matrix multiplication of the blocks themselves.
        # So we need to decompose each block multiplication.
        # So the total rank is 7 (outer) * 7 (inner) = 49.
        
        # So the algorithm is:
        # For each outer term k (7 terms):
        #   Get u2_k, v2_k, w2_k.
        #   For each inner term s (7 terms):
        #     Get u3_s, v3_s, w3_s.
        #     Construct global u, v, w.
        #     For i in 0..3:
        #       coeff = u2_k[i] * u3_s[i]
        #       If coeff != 0:
        #         Add coeff to the 4 entries of block i in u.
        #       Similarly for v and w.
        
        pass

    # Implementation
    terms_4x4 = []
    for k in range(7):
        u2, v2, w2 = strassen_2x2[k]
        for s in range(7):
            u3, v3, w3 = strassen_2x2[s]
            u = [0] * 16
            v = [0] * 16
            w = [0] * 16
            
            for i in range(4):
                coeff_u = u2[i] * u3[i]
                coeff_v = v2[i] * v3[i]
                coeff_w = w2[i] * w3[i]
                
                # Get global indices for block i
                if i == 0:
                    idx = [0, 1, 4, 5]
                elif i == 1:
                    idx = [2, 3, 6, 7]
                elif i == 2:
                    idx = [8, 9, 12, 13]
                else:
                    idx = [10, 11, 14, 15]
                
                for j in range(4):
                    u[idx[j]] = coeff_u
                    v[idx[j]] = coeff_v
                    w[idx[j]] = coeff_w
                    
            terms_4x4.append((u, v, w))
            
    # Now we have 49 terms for 4x4x4.
    # We need 4x4x7.
    # We pad B to 4x8.
    # B is split into B_L (4x4) and B_R (4x4).
    # C = A B_L + A B_R.
    # We use the 49 terms for A B_L.
    # We use the 49 terms for A B_R.
    # Total 98 terms.
    
    terms_4x4x8 = []
    
    # For B_L, the blocks are the same as B in 4x4x4.
    # For B_R, the blocks are shifted?
    # In the 4x4x4 decomposition, B is 4x4.
    # For B_L, we use the same B indices.
    # For B_R, we use the same B indices but they correspond to the right half of the 4x8 matrix.
    # The global B vector for 4x8 has length 32.
    # B_L occupies indices 0-15.
    # B_R occupies indices 16-31.
    
    for (u, v, w) in terms_4x4:
        # Term for B_L
        v_L = [0] * 32
        for i in range(16):
            v_L[i] = v[i]
        # w is for C_L (left half of C), so indices 0-15 of C (4x8)
        w_L = [0] * 32
        for i in range(16):
            w_L[i] = w[i]
        terms_4x4x8.append((u, v_L, w_L))
        
        # Term for B_R
        v_R = [0] * 32
        for i in range(16):
            v_R[16+i] = v[i]
        # w is for C_R (right half of C), so indices 16-31 of C (4x8)
        w_R = [0] * 32
        for i in range(16):
            w_R[16+i] = w[i]
        terms_4x4x8.append((u, v_R, w_R))
        
    # We have 98 terms.
    # U is 16x98.
    # V is 32x98.
    # W is 32x98.
    
    # We need W to be 28x98 (for 4x7 output).
    # So we take the first 28 rows of W.
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    for r in range(98):
        u, v, w = terms_4x4x8[r]
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
    W_trimmed = W[:28, :]
    
    return {"U": U, "V": V, "W": W_trimmed}
