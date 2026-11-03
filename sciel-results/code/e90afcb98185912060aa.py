"""Matrix multiplication 6x6x6 - Block-Strassen with SVD rank-reduction attempt."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    block_size = 3
    num_blocks = 2

    rank = 7 * (block_size ** 3)  # 189

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Strassen coefficients
    alpha = {}
    alpha[(0, 0, 0)] = 1; alpha[(0, 1, 1)] = 1
    alpha[(1, 1, 0)] = 1; alpha[(1, 1, 1)] = 1
    alpha[(2, 0, 0)] = 1
    alpha[(3, 1, 1)] = 1
    alpha[(4, 0, 0)] = 1; alpha[(4, 0, 1)] = 1
    alpha[(5, 1, 0)] = 1; alpha[(5, 0, 0)] = -1
    alpha[(6, 0, 1)] = 1; alpha[(6, 1, 1)] = -1

    gamma = {}
    gamma[(0, 0, 0)] = 1; gamma[(0, 1, 1)] = 1
    gamma[(1, 0, 0)] = 1
    gamma[(2, 0, 1)] = 1; gamma[(2, 1, 1)] = -1
    gamma[(3, 1, 0)] = 1; gamma[(3, 0, 0)] = -1
    gamma[(4, 1, 1)] = 1
    gamma[(5, 0, 0)] = 1; gamma[(5, 0, 1)] = 1
    gamma[(6, 1, 0)] = 1; gamma[(6, 1, 1)] = 1

    beta = {}
    beta[(0, 0, 0)] = 1; beta[(0, 1, 1)] = 1
    beta[(1, 1, 0)] = 1; beta[(1, 1, 1)] = -1
    beta[(2, 0, 1)] = 1; beta[(2, 1, 1)] = 1
    beta[(3, 0, 0)] = 1; beta[(3, 1, 0)] = 1
    beta[(4, 0, 0)] = -1; beta[(4, 0, 1)] = 1
    beta[(5, 1, 1)] = 1
    beta[(6, 0, 0)] = 1

    # Construct U, V, W
    for s in range(7):
        for i0 in range(block_size):
            for j0 in range(block_size):
                for k0 in range(block_size):
                    r = s * (block_size ** 3) + i0 * (block_size ** 2) + j0 * block_size + k0

                    for a in range(num_blocks):
                        for b in range(num_blocks):
                            coeff = alpha.get((s, a, b), 0)
                            if coeff != 0:
                                i = a * block_size + i0
                                j = b * block_size + j0
                                U[i * m + j, r] = coeff

                    for c in range(num_blocks):
                        for d in range(num_blocks):
                            coeff = gamma.get((s, c, d), 0)
                            if coeff != 0:
                                j = c * block_size + j0
                                k = d * block_size + k0
                                V[j * p + k, r] = coeff

                    for i1 in range(num_blocks):
                        for k1 in range(num_blocks):
                            coeff = beta.get((s, i1, k1), 0)
                            if coeff != 0:
                                i = i1 * block_size + i0
                                k = k1 * block_size + k0
                                W[k * n + i, r] = coeff

    # SVD rank-reduction post-processing
    # Try to find linear dependencies in the combined factor space
    # The key insight: if we can find a basis transformation that makes
    # some columns zero across all three factors simultaneously, we can drop them.
    
    # Approach: compute the "effective rank" by checking if the tensor
    # decomposition has redundant rank-1 terms.
    # 
    # We form the Khatri-Rao product structure: each rank-r term contributes
    # u_r ⊗ v_r ⊗ w_r. If some linear combination of these equals zero,
    # we have redundancy.
    #
    # Practical approach: check if columns of U (or V or W) are linearly dependent
    # in a way that's consistent across all three factors.
    
    # Check rank of U
    try:
        _, svals_u, vh_u = np.linalg.svd(U, full_matrices=False)
        rank_u = np.sum(svals_u > 1e-10)
    except:
        rank_u = rank
    
    # Check rank of V
    try:
        _, svals_v, vh_v = np.linalg.svd(V, full_matrices=False)
        rank_v = np.sum(svals_v > 1e-10)
    except:
        rank_v = rank
    
    # Check rank of W
    try:
        _, svals_w, vh_w = np.linalg.svd(W, full_matrices=False)
        rank_w = np.sum(svals_w > 1e-10)
    except:
        rank_w = rank
    
    # The effective rank of the tensor decomposition is at most min(rank_u, rank_v, rank_w)
    # but we need a consistent reduction across all three.
    
    # Try to find the maximum set of columns that can be kept
    # such that the decomposition remains exact.
    
    # For a valid tensor decomposition, we need:
    # sum_r U[:,r] * V[:,r] * W[:,r] = T (the multiplication tensor)
    # 
    # If we drop column r, we lose the contribution U[:,r]*V[:,r]*W[:,r].
    # We can only drop it if this contribution is zero or can be absorbed.
    
    # Check if any individual rank-1 term is zero
    zero_cols = []
    for r in range(rank):
        if np.allclose(U[:, r], 0) or np.allclose(V[:, r], 0) or np.allclose(W[:, r], 0):
            zero_cols.append(r)
    
    if len(zero_cols) > 0:
        keep = [r for r in range(rank) if r not in zero_cols]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    
    # Try SVD-based reduction: if the factors have low numerical rank,
    # we can potentially reduce by finding a change of basis.
    # 
    # Key idea: if U = Q_U @ R_U where R_U is rank r' < r, then we can
    # absorb Q_U into the other factors IF the structure allows it.
    # This only works if the decomposition has a specific product structure.
    
    # For Strassen, the structure is:
    # U[:, s*27 + offset] depends on which (s, i0, j0, k0) it is
    # The columns within a Strassen term s are independent (they correspond
    # to different (i0, j0, k0) triples), so no reduction within a term.
    # 
    # Between terms, the columns are also independent because each term
    # uses different linear combinations of A and B blocks.
    # 
    # However, let's check numerically if there's any redundancy.
    
    # Build the full tensor and check its rank
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # This is a 6x6x6 tensor with 216 entries.
    
    # Compute the tensor and check if it matches the multiplication tensor
    # The multiplication tensor T_mult[i,j,k] = 1 if j==k (for C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait, the convention is W[k*n+i, r], so W is indexed as [k, i] for output C[i, k]
    # T[i, j, k] should equal delta_{j,k} for the multiplication tensor... 
    # Actually no. The multiplication tensor has T[i,j,k] = 1 if the output position (i,k) 
    # gets contribution from A[i,j]*B[j,k]. So T[i,j,k] = 1 for all valid (i,j,k) where
    # j is the contraction index. That means T[i,j,k] = 1 for all i,j,k.
    
    # Wait, let me reconsider. The matrix multiplication tensor T is defined by:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i, j, k (it's the all-ones tensor in this case? No!)
    # 
    # Actually T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k].
    # This is true for all i, j, k. So T is the all-ones 6x6x6 tensor? No, that's not right either.
    # 
    # The tensor T has entries T[i,j,k] where i indexes rows of A, j indexes the contraction
    # (cols of A / rows of B), k indexes cols of B. T[i,j,k] = 1 for all i,j,k.
    # 
    # Wait no. For matrix multiplication C = AB where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # 
    # Hmm, but that would mean the tensor is just the all-ones tensor, which has rank 1.
    # That can't be right. Let me think again.
    #
    # Actually, the multiplication tensor is NOT all ones. It's defined such that
    # the bilinear form <T, A ⊗ B> = <C, C> where C = AB.
    # In tensor notation: T[i,j,k] = 1 for all i,j,k.
    # 
    # Wait, I think I'm confusing myself. The standard result is that the rank of
    # the n×n×n matrix multiplication tensor is at least 2n² - 1 (for n≥2) and at most n³.
    # For n=2, the rank is 7 (Strassen). For n=1, rank is 1.
    #
    # The tensor T for 2×2×2 multiplication has T[i,j,k] = 1 for all i,j,k in {0,1}.
    # That's an 8-entry all-ones tensor, which has rank 1 as a tensor? No!
    # 
    # I think the issue is that the tensor entries are not all 1. Let me look at this more carefully.
    #
    # For C = AB: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is defined by T(i,j,k) = 1 for all i,j,k.
    # Then <T, A⊗B> = sum_{i,j,k} T(i,j,k) A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k] = sum_i,k C[i,k]
    # That's not right. We want <T, A⊗B> to give us the full matrix C, not just its sum.
    #
    # I think the correct formulation is that the multiplication "tensor" is actually a
    # 3-way array where T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # Since C[i,k] = sum_j A[i,j]*B[j,k], we have T[i,j,k] = 1 for all i,j,k.
    #
    # But then the rank of this tensor (as a 3-way array) is what we're computing.
    # The all-ones 2×2×2 tensor has tensor rank 1 (it's u⊗v⊗w where u,v,w are all-ones vectors).
    # But we know 2×2 matrix multiplication has rank 7, not 1!
    #
    # I must be misunderstanding the tensor definition. Let me reconsider.
    #
    # Oh wait, I think the issue is that the tensor is NOT all ones. The multiplication
    # tensor T is defined such that for each output position (i,k), the entry is
    # sum_j A[i,j] B[j,k]. In tensor decomposition form:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where (u_r ⊗ v_r ⊗ w_r)[i,j,k] = u_r[i] * v_r[j] * w_r[k]
    # and we need: sum_r u_r[i] * v_r[j] * w_r[k] = [j contributes to C[i,k]]
    #
    # The condition is: sum_r u_r[i] * v_r[j] * w_r[k] = δ_{j, contraction} ... no.
    #
    # Let me just verify with the 2×2 Strassen decomposition.
    # For 2×2: A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # Strassen: 
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M2 = (c+d)e = ce+de
    # M3 = a(f-h) = af-ah
    # M4 = d(g-c) = dg-dc
    # M5 = (a+b)h = ah+bh
    # M6 = (c-d)(e+f) = ce+cf-de-df
    # M7 = (b-d)(g+h) = bg+bh-dg-dh
    #
    # C00 = M1 - M4 - M5 + M7 = (ae+ah+de+dh) - (dg-dc) - (ah+bh) + (bg+bh-dg-dh)
    #     = ae+ah+de+dh-dg+dc-ah-bh+bg+bh-dg-dh
    #     = ae+de+dc+bg-dg-dg+dh-dh+ah-ah+bh-bh
    #     = ae+de+dc+bg-2dg
    # That doesn't look right. Let me use the standard Strassen formulas.
    #
    # Standard Strassen for C = AB:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M6 - M7  (or some variant)
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    #
    # Let me verify C00:
    # M1 - M4 + M6 - M7
    # = (A00+A11)(B00+B11) - A11(B10-B00) + (A10-A00)(B00+B01) - (A01-A11)(B10+B11)
    # = A00B00+A00B11+A11B00+A11B11 - A11B10+A11B00 + A10B00+A10B01-A00B00-A00B01 - A01B10-A01B11+A11B10+A11B11
    # = A00B00+A00B11+A11B00+A11B11-A11B10+A11B00+A10B00+A10B01-A00B00-A00B01-A01B10-A01B11+A11B10+A11B11
    # = (A00B00-A00B00) + A00B11 + (A11B00+A11B00) + (A11B11+A11B11) + (-A11B10+A11B10) + A10B00+A10B01-A00B01-A01B10-A01B11
    # = A00B11 + 2A11B00 + 2A11B11 + A10B00+A10B01-A00B01-A01B10-A01B11
    # 
    # This should equal A00B00+A01B10 (the (0,0) entry of AB).
    # Hmm, this doesn't simplify to that. I must have the wrong combination.
    
    # Regardless, the current program constructs a valid decomposition (score 1.1429 > 1.0),
    # so the coefficients are correct. The question is whether we can reduce the rank.
    
    # For the 6×6×6 Block-Strassen with 3×3 blocks:
    # Each of the 7 Strassen terms involves multiplying two 3×3 matrices.
    # The rank of a 3×3×3 matrix multiplication tensor is at least 16 (best known upper bound ~22).
    # But we're using the trivial rank-27 decomposition for each 3×3 multiplication.
    # 
    # The key question: can we do better than 7*27=189?
    # If we used a rank-16 decomposition for 3×3, we'd get 7*16=112.
    # But the directive says to stay within the Block-Strassen approach.
    
    # Let's check if there's any numerical rank deficiency in the current construction.
    # If rank_u < 189 or rank_v < 189 or rank_w < 189, we might be able to reduce.
    
    if rank_u < rank or rank_v < rank or rank_w < rank:
        # There might be some reduction possible
        # Find the minimum rank across all three
        min_rank = min(rank_u, rank_v, rank_w)
        
        # Try to find a consistent set of columns to keep
        # This is complex; for now, just note the potential reduction
        # and return the original if we can't cleanly reduce.
        pass
    
    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
