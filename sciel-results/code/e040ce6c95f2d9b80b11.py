"""Matrix multiplication 2x4 by 4x8: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Decompose 2x4 * 4x8 as two independent 1x4 * 4x8 row-vector multiplications.
    # For each row, use a rank-3 bilinear scheme (analogous to small Winograd / 
    # Strassen-style decomposition for the 4x8 convolution-like product).
    #
    # The product of a 1x4 row vector a = [a0,a1,a2,a3] with a 4x8 matrix B
    # (rows b0..b3, each of length 8) gives an 1x8 result:
    #   c_k = a0*b0_k + a1*b1_k + a2*b2_k + a3*b3_k,  k=0..7
    #
    # We seek a bilinear decomposition of rank r per row:
    #   c_k = sum_{t=1}^{r} U_t * V_t
    # where U_t = sum_i alpha_{t,i} * a_i  (linear form in a)
    #       V_t = sum_k beta_{t,k} * (row of B applied to produce a scalar per k)
    #
    # Actually, for matrix-vector: c = a^T B, where B is 4x8.
    # c_k = sum_i a_i * B[i,k]
    # This is a bilinear form in a (dim 4) and B (dim 32).
    #
    # A standard approach: use the fact that 1x4 x 4x8 can be seen as
    # multiplying a 4-vector by a 4x8 matrix. We can decompose using
    # rank-3 if we allow the "V" side to be linear combinations of columns.
    #
    # Let's use a concrete rank-3 scheme per row.
    # For a 1x4 times 4x8:
    #   c_k = sum_{i=0}^{3} a_i * B[i,k]
    #
    # Define 3 intermediate products:
    #   p1 = a0 * (B[0,k] + B[1,k])   -- but this doesn't work directly.
    #
    # Alternative: use the fact that we can group terms.
    # Consider:
    #   p1 = (a0 + a1) * (B[0,k] + B[1,k])  -> gives a0*B[0,k] + a1*B[1,k] + a0*B[1,k] + a1*B[0,k]
    #   p2 = (a0 - a1) * (B[0,k] - B[1,k])  -> gives a0*B[0,k] - a1*B[1,k] - a0*B[1,k] + a1*B[0,k]
    #   p3 = (a2 + a3) * (B[2,k] + B[3,k])
    #   p4 = (a2 - a3) * (B[2,k] - B[3,k])
    #
    # This is rank 4, not 3. Let me reconsider.
    #
    # For a true rank-3 decomposition of 1x4 x 4x8, we need:
    #   c_k = sum_{t=1}^{3} u_t * v_{t,k}
    # where u_t = sum_i alpha_{t,i} a_i and v_{t,k} = sum_i beta_{t,i,k} B[i,k]
    #
    # This requires: for all i,k: delta_{i,k} = sum_t alpha_{t,i} * beta_{t,i',k}... 
    # Actually the tensor is T[i,k,t] where the multiplication tensor for 1x4*4x8 has
    # entries T[a_index, b_index, c_index]. The b_index is (row, col) = (i,k) with i in 0..3, k in 0..7.
    #
    # The tensor is: T[i, (j,k)] = 1 if i==j else 0, for the product c_k = sum_i a_i B[i,k].
    # Wait, that's just the identity in the i dimension. So the tensor is:
    #   T[i, j, k] = 1 if i == j, for i,j in 0..3, k in 0..7
    # This is a block-diagonal tensor: 8 copies of the 4x4 identity.
    # The rank of the 4x4 identity tensor is 4 (since it's the identity matrix, its tensor rank is 4).
    # So the rank of 1x4 * 4x8 is 8 * 4 / 8... no.
    #
    # The tensor for c_k = sum_i a_i B[i,k] has mode sizes 4 (for a), 32 (for B entries), 8 (for c).
    # T[i, (j,k), l] = 1 if i==j and l==k, else 0.
    # This is sum_{i=0}^{3} sum_{k=0}^{7} e_i ⊗ e_{i,k} ⊗ e_k
    # So the rank is at most 4*8 = 32, but we can do better.
    #
    # Actually, for fixed k, the sub-tensor is the 4x4 identity, which has rank 4.
    # For 8 different k values, if they were independent, rank would be 8*4=32.
    # But we can share: the structure is T = sum_{k=0}^{7} I_4 ⊗ e_k (in appropriate modes).
    # The rank of I_4 is 4, and since the k-dimension is separate, the total rank is 8*4=32? 
    # No, that's the trivial bound. The actual rank might be lower.
    #
    # Let me just use a simple approach: for each row of A, use rank 8 (one per output element).
    # Total rank = 2 * 8 = 16. That's already better than 64.
    #
    # But we can do better. For 1x4 * 4x8, note that c_k = a^T B[:,k].
    # This is just 8 dot products of a (dim 4) with columns of B.
    # The tensor rank of a 1x4 x 4x8 multiplication: 
    #   T[a, B, c] where a is 4-dim, B is 32-dim, c is 8-dim.
    #   T[i, (j,k), l] = delta_{i,j} * delta_{k,l}
    # This factors as (I_4 in a-B_row modes) ⊗ (I_8 in B_col-c modes).
    # The rank of a Kronecker product of tensors is the product of ranks.
    # Rank(I_4 tensor) = 4, Rank(I_8 tensor) = 8.
    # So rank = 4 * 8 = 32? That seems high.
    #
    # Actually, the tensor T[i, jk, l] = delta_{ij} delta_{kl} can be written as:
    #   sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    # where u_t is in R^4, v_t in R^32, w_t in R^8.
    # We need: delta_{ij} delta_{kl} = sum_t u_t[i] v_t[jk] w_t[l]
    #
    # One decomposition: for each (i,k) pair, use one term: u = e_i, v = e_{i,k}, w = e_k.
    # That gives rank 4*8 = 32.
    #
    # Can we do better? The tensor is the Kronecker product of two identity matrices.
    # The rank of the Kronecker product of two matrices (as tensors) is the product of their ranks.
    # Rank of 4x4 identity as a tensor (3-way, but really it's a 2-way tensor / matrix) is 4.
    # Rank of 8x8 identity is 8.
    # So the Kronecker product has rank 4*8 = 32.
    #
    # Hmm, so rank 32 per row? That gives total rank 64, same as schoolbook.
    #
    # Wait, I think I'm confusing things. Let me reconsider the problem.
    #
    # The matrix multiplication C = A*B where A is 2x4, B is 4x8, C is 2x8.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has modes: A (2*4=8 entries), B (4*8=32 entries), C (2*8=16 entries).
    # T[a_idx, b_idx, c_idx] = 1 if a_idx corresponds to (i,j), b_idx to (j,k), c_idx to (i,k).
    #
    # The schoolbook rank is 2*4*8 = 64.
    #
    # For 1x4 * 4x8: rank is at most 4*8=32 (trivial), but can it be lower?
    # The tensor for 1x4 * 4x8: T[j, (j',k), k'] where j is the A index (0..3), 
    # (j',k) is the B index, k' is the C index. T = 1 if j==j' and k==k'.
    # This is the same as before: rank 32.
    #
    # So two independent rows give rank 64. No improvement.
    #
    # Let me try a different approach. Use the fact that A has only 2 rows.
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    #
    # We can write this as: for each k, C[:,k] = A * B[:,k]
    # where A is 2x4 and B[:,k] is 4x1.
    #
    # The tensor for 2x4 * 4x1 (matrix-vector) has rank at most 2*4=8.
    # For 8 different k values, if independent, rank 8*8=64.
    # But we can share structure.
    #
    # Actually, let's just use the trivial decomposition but optimized:
    # For each (i,j,k): one term. Total 2*4*8 = 64 terms.
    #
    # Since I can't easily find a lower rank, let me just return the schoolbook with rank 64.
    # But wait, the observed change shows a score of 1.142857... which is 8/7, suggesting rank 49?
    # 64/49 ≈ 1.306, not 1.142. 64/56 ≈ 1.142. So rank 56?
    #
    # Let me try to get rank 56 or lower. 
    # 
    # For 2x4 * 4x8, one approach: use the fact that the 4x8 matrix B can be viewed as
    # 8 columns of 4-vectors. The product is 2x8.
    #
    # Alternative: use a block decomposition. Split B into two 4x4 blocks: B = [B1 | B2].
    # C = A * B1 (2x4) concatenated with A * B2 (2x4).
    # Each 2x4 * 4x4 has schoolbook rank 32. Total 64. No help.
    #
    # Let me just implement the schoolbook and see. The task says lower is better, 
    # and the seed is rank 64. Any improvement helps.
    #
    # Actually, I realize I should just try to find ANY valid decomposition with rank < 64.
    # Let me use a simple rank-64 decomposition (schoolbook) as a fallback, but try to
    # find something better.
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Schoolbook: rank = n*m*p = 64
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
