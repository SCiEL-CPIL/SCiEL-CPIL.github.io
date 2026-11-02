"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Verify that U @ diag(1) @ V^T @ ... reconstructs the tensor."""
    n, m, p = 2, 4, 4
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k,l] = sum_r U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r]
    # Build tensor
    U3 = U.reshape(n, m, -1)  # (n, m, rank)
    V3 = V.reshape(m, p, -1)  # (m, p, rank)
    W3 = W.reshape(p, n, -1)  # (p, n, rank)
    # T[i,j,k,l] = sum_r U3[i,j,r] * V3[k,l,r] * W3[l,i,r]
    # Use einsum
    T = np.einsum('ijr,klr,lir->ijkl', U3, V3, W3)
    # Expected: T[i,j,k,l] = 1 if j==k else 0
    expected = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                expected[i, j, j, l] = 1.0
    return np.allclose(T, expected, atol=1e-9)

def _strassen_block_UVW(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base, m, p, n):
    """Return list of (U_idx, V_idx, W_idx, val) tuples for one Strassen block."""
    a00_u = a_row_base * m + a_col_base
    a01_u = a_row_base * m + a_col_base + 1
    a10_u = (a_row_base + 1) * m + a_col_base
    a11_u = (a_row_base + 1) * m + a_col_base + 1
    
    b00_v = b_row_base * p + b_col_base
    b01_v = b_row_base * p + b_col_base + 1
    b10_v = (b_row_base + 1) * p + b_col_base
    b11_v = (b_row_base + 1) * p + b_col_base + 1
    
    c00_w = c_col_base * n + c_row_base
    c01_w = (c_col_base + 1) * n + c_row_base
    c10_w = c_col_base * n + (c_row_base + 1)
    c11_w = (c_col_base + 1) * n + (c_row_base + 1)
    
    terms = []
    # M1 = (a00+a11)(b00+b11)
    terms.append((a00_u, b00_v, c00_w, 1))
    terms.append((a00_u, b11_v, c00_w, 1))
    terms.append((a11_u, b00_v, c00_w, 1))
    terms.append((a11_u, b11_v, c00_w, 1))
    terms.append((a00_u, b00_v, c11_w, 1))
    terms.append((a00_u, b11_v, c11_w, 1))
    terms.append((a11_u, b00_v, c11_w, 1))
    terms.append((a11_u, b11_v, c11_w, 1))
    
    # M2 = (a10+a11)b00
    terms.append((a10_u, b00_v, c10_w, 1))
    terms.append((a11_u, b00_v, c10_w, 1))
    terms.append((a10_u, b00_v, c11_w, -1))
    terms.append((a11_u, b00_v, c11_w, -1))
    
    # M3 = a00(b01-b11)
    terms.append((a00_u, b01_v, c01_w, 1))
    terms.append((a00_u, b11_v, c01_w, -1))
    terms.append((a00_u, b01_v, c11_w, 1))
    terms.append((a00_u, b11_v, c11_w, -1))
    
    # M4 = a11(b10-b00)
    terms.append((a11_u, b10_v, c00_w, 1))
    terms.append((a11_u, b00_v, c00_w, -1))
    terms.append((a11_u, b10_v, c10_w, 1))
    terms.append((a11_u, b00_v, c10_w, -1))
    
    # M5 = (a00+a01)b11
    terms.append((a00_u, b11_v, c00_w, -1))
    terms.append((a01_u, b11_v, c00_w, -1))
    terms.append((a00_u, b11_v, c01_w, 1))
    terms.append((a01_u, b11_v, c01_w, 1))
    
    # M6 = (a10-a00)(b00+b01)
    terms.append((a10_u, b00_v, c11_w, 1))
    terms.append((a00_u, b00_v, c11_w, -1))
    terms.append((a10_u, b01_v, c11_w, 1))
    terms.append((a00_u, b01_v, c11_w, -1))
    
    # M7 = (a01-a11)(b10+b11)
    terms.append((a01_u, b10_v, c00_w, 1))
    terms.append((a11_u, b10_v, c00_w, -1))
    terms.append((a01_u, b11_v, c00_w, 1))
    terms.append((a11_u, b11_v, c00_w, -1))
    
    return terms

def construct():
    n, m, p = 2, 4, 4
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Start with the known rank-28 Strassen decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        # M1 = (a00+a11)(b00+b11)
        U[a00_u][offset + 0] += 1
        U[a11_u][offset + 0] += 1
        V[b00_v][offset + 0] += 1
        V[b11_v][offset + 0] += 1
        W[c00_w][offset + 0] += 1
        W[c11_w][offset + 0] += 1
        
        # M2 = (a10+a11)b00
        U[a10_u][offset + 1] += 1
        U[a11_u][offset + 1] += 1
        V[b00_v][offset + 1] += 1
        W[c10_w][offset + 1] += 1
        W[c11_w][offset + 1] -= 1
        
        # M3 = a00(b01-b11)
        U[a00_u][offset + 2] += 1
        V[b01_v][offset + 2] += 1
        V[b11_v][offset + 2] -= 1
        W[c01_w][offset + 2] += 1
        W[c11_w][offset + 2] += 1
        
        # M4 = a11(b10-b00)
        U[a11_u][offset + 3] += 1
        V[b10_v][offset + 3] += 1
        V[b00_v][offset + 3] -= 1
        W[c00_w][offset + 3] += 1
        W[c10_w][offset + 3] += 1
        
        # M5 = (a00+a01)b11
        U[a00_u][offset + 4] += 1
        U[a01_u][offset + 4] += 1
        V[b11_v][offset + 4] += 1
        W[c00_w][offset + 4] -= 1
        W[c01_w][offset + 4] += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10_u][offset + 5] += 1
        U[a00_u][offset + 5] -= 1
        V[b00_v][offset + 5] += 1
        V[b01_v][offset + 5] += 1
        W[c11_w][offset + 5] += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01_u][offset + 6] += 1
        U[a11_u][offset + 6] -= 1
        V[b10_v][offset + 6] += 1
        V[b11_v][offset + 6] += 1
        W[c00_w][offset + 6] += 1
    
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    # Verify rank-28 is correct
    assert _verify(U, V, W), "Rank-28 Strassen decomposition failed verification"
    
    # Try to find rank-27 by attempting to merge/eliminate one term
    # Strategy: try removing one rank-1 term and see if the remaining 27 can be adjusted
    # to still span the tensor. Use a linear algebra approach.
    
    # The tensor T has entries T[i,j,k,l] = delta_{j,k}
    # We need: sum_r U[i,j,r] * V[k,l,r] * W[l,i,r] = delta_{j,k}
    
    # Try: can we express one of the 28 rank-1 terms as a linear combination
    # of the other 27? If so, we can eliminate it.
    
    # Alternative: try a different factorization structure.
    # Key idea: The 4 Strassen blocks share the same A rows. 
    # Block 0: A cols 0,1; B rows 0,1; C cols 0,1
    # Block 1: A cols 2,3; B rows 0,1; C cols 0,1
    # Block 2: A cols 0,1; B rows 2,3; C cols 0,1
    # Block 3: A cols 2,3; B rows 2,3; C cols 0,1
    
    # Since A has only 2 rows, the U vectors for different blocks overlap in row indices.
    # This might allow sharing.
    
    # Let's try: use a 2x2 Strassen on the (A-row, B-row) plane with 4-column B.
    # Actually, let me try a completely different approach:
    # View as: for each output column l, C[:,l] = A @ B[:,l]
    # This is 4 separate 2x4 @ 4x1 multiplications.
    # But that's not helpful directly.
    
    # Let me try: rank-27 via a specific algebraic identity.
    # Consider combining blocks 0 and 2 (same A columns 0,1, different B rows).
    # And blocks 1 and 3 (same A columns 2,3, different B rows).
    
    # For fixed A columns (j0, j1), we need to compute:
    # C[i,l] = A[i,j0]*B[j0,l] + A[i,j1]*B[j1,l] for l in {0,1,2,3}
    # This is a 2x4 @ 4x4 restricted to 2 B-rows = 2x4 @ 2x4 (for each pair of B-rows)
    
    # Actually for fixed A columns j0,j1 and fixed B rows k0,k1:
    # C[i,l] += A[i,j0]*B[k0,l] + A[i,j1]*B[k1,l] ... no that's not right.
    
    # Let me just try to verify if rank 27 is achievable by a direct construction.
    # I'll attempt a known result: M_{2,4,4} has rank 27.
    
    # Construction idea: Use the fact that we can write the tensor as:
    # T = sum over 27 rank-1 terms
    # 
    # Let me try: 3 full Strassen blocks (21 terms) + 6 additional terms for the 4th block
    # with one term shared/absorbed.
    
    # Actually, let me try a different decomposition structure entirely.
    # Split B into 4 single columns. For each column l of B, we compute A @ B[:,l].
    # A @ B[:,l] is 2x4 @ 4x1 = 2x1.
    # The tensor for A @ B[:,l] has rank at most 2*4 = 8, but actually it's a rank-4 tensor
    # (since it's just outer product structure).
    
    # Hmm, let me try yet another approach. Use the following identity:
    # For 2x4 @ 4x4, we can use a "bordered" Strassen approach.
    
    # Let me try to construct rank-27 explicitly.
    # 
    # Define 27 rank-1 terms. The key is to find U, V, W matrices of rank 27.
    #
    # Approach: Start with the 4 Strassen blocks but modify one block to use 6 terms
    # instead of 7, by exploiting that the A-part of that block can be combined with
    # terms from another block.
    
    # Let me try: for the last Strassen block (cols 2,3 of A; rows 2,3 of B; cols 2,3 of C),
    # use only 6 terms and compensate in the other blocks.
    
    # This is getting complex. Let me try a numerical search approach.
    
    if time.time() < deadline:
        # Try to find rank-27 via iterative improvement
        # Start from rank-28 and try to zero out one column
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each candidate column r to eliminate, check if the tensor
        # can be represented with rank 27.
        # This is equivalent to: is the r-th rank-1 term in the span of the others?
        # In the tensor space, this means: is u_r ⊗ v_r ⊗ w_r in the span of
        # {u_s ⊗ v_s ⊗ w_s : s ≠ r}?
        
        # A simpler check: try to solve for coefficients.
        # The tensor T = sum_r u_r ⊗ v_r ⊗ w_r
        # We want to know if T = sum_{s≠r} alpha_s * (u_s ⊗ v_s ⊗ w_s) for some alpha_s
        # But that's not quite right either, since we can change the u_s, v_s, w_s.
        
        # Better approach: try to find a rank-27 decomposition from scratch using
        # a structured search.
        
        # Let me try a specific rank-27 construction based on algebraic identities.
        # 
        # Key insight: For M_{2,4,4}, we can use the following structure:
        # - 4 terms for the "diagonal" (A[i,j]*B[j,l]*C[i,l])
        # - Additional terms for off-diagonal corrections
        #
        # Actually, the schoolbook is 32 terms. Strassen gives 28. 
        # The question is whether 27 is achievable.
        
        # Let me try a direct construction with 27 terms.
        # 
        # I'll use the following approach:
        # For the 2x2 subblocks, use a modified Strassen that exploits
        # the shared A dimension.
        
        # Let me define the 27 terms explicitly:
        # Terms 0-6: Strassen block for A[0:2,0:2], B[0:2,0:2], C[0:2,0:2]
        # Terms 7-13: Strassen block for A[0:2,2:4], B[0:2,0:2], C[0:2,0:2]
        # Terms 14-20: Strassen block for A[0:2,0:2], B[2:4,0:2], C[0:2,0:2]
        # Terms 21-26: Modified block for A[0:2,2:4], B[2:4,0:2], C[0:2,0:2] (6 terms)
        
        # The last block needs to compute:
        # C[i,l] += A[i,2]*B[2,l] + A[i,3]*B[3,l] for i in {0,1}, l in {0,1}
        # 
        # With 6 terms instead of 7, we need to find a 6-term formula for
        # (2x2) @ (2x2) -> (2x2) where the A-matrix entries are shared with
        # other blocks.
        
        # Actually, this won't work directly because each 2x2@2x2->2x2 needs
        # at least 7 terms (Strassen is optimal for 2x2).
        
        # The trick must be in sharing terms ACROSS blocks.
        
        # Let me try: can we combine terms from different blocks?
        # E.g., term M1 of block 0 uses U[a00], U[a11], V[b00], V[b11], W[c00], W[c11]
        # Term M1 of block 1 uses U[a00], U[a11] (same A indices!), V[b20], V[b21], W[c00], W[c11]
        
        # The U vectors for a00 and a11 appear in both block 0 and block 1.
        # This means the U-matrix has non-zero entries in multiple columns for the same row.
        # This is fine - it's just how the decomposition works.
        
        # The question is whether we can reduce the total number of columns.
        
        # Let me try a completely different factorization.
        # 
        # Use the following identity for 2x4 @ 4x4:
        # Let A = [A1 | A2] where A1, A2 are 2x2
        # Let B = [B1; B2] where B1, B2 are 2x4 (split by rows)
        # Then C = A1*B1 + A2*B2
        # 
        # A1*B1 is 2x2 @ 2x4 = 2x4, rank at most 2*2*4=16, but with Strassen-like:
        # Split B1 = [B1a | B1b] where B1a, B1b are 2x2
        # A1*B1a: 2x2 @ 2x2 -> 2x2, rank 7 (Strassen)
        # A1*B1b: 2x2 @ 2x2 -> 2x2, rank 7 (Strassen)
        # A2*B1a: 2x2 @ 2x2 -> 2x2, rank 7
        # A2*B1b: 2x2 @ 2x2 -> 2x2, rank 7
        # Total: 28. Same as before.
        
        # The key question: can we do better by NOT splitting independently?
        
        # Let me try: A1*(B1a+B1b) + A2*(B1a+B1b) = (A1+A2)*(B1a+B1b) ... no that's wrong.
        
        # C = A1*B1 + A2*B2
        # = A1*(B1) + A2*(B2)
        # 
        # What if we compute (A1+A2)*(B1+B2) and correct?
        # (A1+A2)*(B1+B2) = A1*B1 + A1*B2 + A2*B1 + A2*B2
        # We want A1*B1 + A2*B2, so we need to subtract A1*B2 + A2*B1.
        # That's 7 + 7 + 7 + 7 = 28 again (or more).
        
        # I think 28 might actually be optimal for this specific case, or 27 requires
        # a very clever construction.
        
        # Let me try one more thing: use a 2x4 @ 4x2 subproblem twice.
        # C[:, 0:2] = A @ B[:, 0:2] (2x4 @ 4x2 = 2x2)
        # C[:, 2:4] = A @ B[:, 2:4] (2x4 @ 4x2 = 2x2)
        # 
        # For 2x4 @ 4x2: rank is known to be 6 (not 7).
        # Reference: The bilinear complexity of M_{2,4,2} is 6.
        # 
        # So total rank = 6 + 6 = 12? That seems too low. Let me check.
        # 
        # Wait, M_{2,4,2} means multiplying 2x4 by 4x2 to get 2x2.
        # The rank of this tensor: lower bound is max(2*4, 4*2, 2*2) = 8.
        # Upper bound: schoolbook is 2*4*2 = 16.
        # I'm not sure of the exact rank. Let me not assume.
        
        # Let me just return the rank-28 solution which is verified.
        # If I can't find rank-27 in time, rank-28 is safe.
        
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
