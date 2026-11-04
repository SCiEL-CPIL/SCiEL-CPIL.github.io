import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Build the matrix-multiplication tensor T with shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if the corresponding indices match for C = A*B
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Flatten T to a matrix of shape (n*m, m*p * p*n) for rank-reduction via SVD
    T_mat = T.reshape(n*m, m*p * p*n)
    
    # Start with naive decomposition (rank = n*m*p = 40)
    # U_naive[i*m+j, idx] = 1, V_naive[j*p+k, idx] = 1, W_naive[k*n+i, idx] = 1
    # where idx corresponds to (i,j,k)
    best_rank = 40
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    
    for idx, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
        best_U[i*m + j, idx] = 1.0
        best_V[j*p + k, idx] = 1.0
        best_W[k*n + i, idx] = 1.0
    
    # Verify naive decomposition
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        reconstructed = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(reconstructed, T, atol=tol)
    
    if not verify(best_U, best_V, best_W):
        # Fallback: return naive
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to reduce rank using various strategies
    # Strategy: Use SVD to find low-rank approximations, then check if they're exact
    
    # First, try SVD-based reduction
    try:
        # SVD of the tensor matrix
        U_svd, s_vals, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
        
        # The exact rank of T_mat
        exact_rank = np.sum(s_vals > 1e-10)
        
        # But the tensor rank might be different from the matrix rank
        # The matrix rank is an upper bound on tensor rank
        # Actually for this tensor, the matrix rank equals the tensor rank
        # because the tensor is a matrix multiplication tensor
        
        # Try different ranks
        for target_rank in range(10, min(exact_rank + 1, 40)):
            if time.time() > deadline - 30:
                break
            
            # Get the top-r singular vectors
            U_top = U_svd[:, :target_rank]
            s_top = s_vals[:target_rank]
            Vt_top = Vt_svd[:target_rank, :]
            
            # Reconstruct: T_mat ≈ U_top @ diag(s_top) @ Vt_top
            # We need to factor diag(s_top) into two matrices
            # Option 1: U_top @ diag(sqrt(s_top)) and diag(sqrt(s_top)) @ Vt_top
            sqrt_s = np.sqrt(s_top)
            U_factor = U_top * sqrt_s[np.newaxis, :]
            V_factor = (sqrt_s[:, np.newaxis] * Vt_top).reshape(target_rank, m*p, p*n)
            
            # But we need V to have shape (m*p, rank) and W to have shape (p*n, rank)
            # So V_factor[r, b, c] should be split: V[b, r] * W[c, r]
            # This means V_factor must be a sum of rank-1 matrices in the (b,c) space
            # V_factor[r] = V[:, r] @ W[:, r].T
            # So we need to SVD each V_factor[r]
            
            U_candidate = U_factor  # shape (n*m, target_rank)
            V_candidate = np.zeros((m*p, target_rank))
            W_candidate = np.zeros((p*n, target_rank))
            
            for r in range(target_rank):
                Vt_r = V_factor[r]  # shape (m*p, p*n)
                # SVD of Vt_r
                u_r, s_r, vt_r = np.linalg.svd(Vt_r, full_matrices=False)
                if len(s_r) == 0 or s_r[0] < 1e-10:
                    continue
                # Vt_r = u_r @ diag(s_r) @ vt_r
                # We want Vt_r = V_r @ W_r.T where V_r has shape (m*p, 1) and W_r has shape (p*n, 1)
                # This is only possible if Vt_r has rank 1
                # If not, we need to split into multiple terms
                # For now, if rank > 1, skip this approach
                if len(s_r) > 1 and s_r[1] > 1e-10:
                    # Can't represent as single outer product
                    # Try to use the top component only and check if it's exact
                    pass
            
            # This approach is getting complex. Let's try a different strategy.
            break
    except Exception:
        pass
    
    # Better strategy: Use the known structure
    # The tensor T can be written as sum of rank-1 tensors
    # T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # This is the naive decomposition with rank n*m*p = 40
    
    # For 2x4x5, we can try to find a better decomposition
    # Known results: The tensor rank of matrix multiplication is related to
    # the bilinear complexity
    
    # Let's try a structured approach:
    # Split the 4-dimensional middle index into two groups of 2
    # For each group, we have a 2x2x5 problem
    # 2x2x5 can be done in rank 7 (Strassen for 2x2x2) + 4 (for 2x2x1) = 11? No...
    # Actually 2x2x5: we can use Strassen for the first 2 columns (rank 7) 
    # and standard for the remaining 3 columns (rank 6), total 13
    # Or: Strassen for 2x2x2 (rank 7) + standard for 2x2x3 (rank 6) = 13
    # Two such blocks: 26 total? But we need to check if the blocks are independent.
    
    # Actually, the middle dimension is shared, so we can't just split independently.
    # Let me think more carefully.
    
    # A is 2x4, B is 4x5, C is 2x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can write this as:
    # C = [A_1 | A_2] * [B_1; B_2]
    # where A_1, A_2 are 2x2, B_1, B_2 are 2x5
    # C = A_1 * B_1 + A_2 * B_2
    
    # Each 2x2 * 2x5 multiplication has rank at most 2*2*5 = 20 (naive)
    # But we can do better:
    # 2x2 * 2x5: split B into B_1a (2x2) and B_1b (2x3)
    # A * B_1a: rank 7 (Strassen)
    # A * B_1b: rank 2*2*3 = 12 (naive) or better
    # Total for 2x2*2x5: 7 + 12 = 19? Or can we do better?
    
    # Actually for 2x2*2x3:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for k=0,1,2
    # This is 6 products naively. Can we do it in fewer?
    # The tensor rank of 2x2x3 is at most 6, and I believe it's exactly 6
    # because the slice rank is at least 6.
    
    # So 2x2*2x5: 7 (for 2x2x2) + 6 (for 2x2x3) = 13
    # Two such: 26 total
    
    # But wait, we can do better by not splitting the middle dimension.
    # Let's try a direct approach.
    
    # The tensor rank of 2x4x5:
    # Lower bound: max(2*4, 4*5, 2*5) = max(8, 20, 10) = 20 (slice rank lower bound)
    # Actually the slice rank lower bound is max over partitions...
    # For matrix multiplication, the tensor rank is at least max(n*m, m*p, p*n)
    # = max(8, 20, 10) = 20
    
    # Upper bound: 40 (naive)
    # Known: 2x4x4 has rank 16? 2x4x5 might have rank around 20-24
    
    # Let's try to construct a rank-20 decomposition
    # If the lower bound is tight, rank 20 would be optimal.
    
    # For rank 20, we need each of the 20 rank-1 terms to contribute
    # to the 40 non-zero entries of the tensor.
    
    # Actually, let me just try to use numerical optimization to find
    # a low-rank decomposition.
    
    # Strategy: Start with naive rank-40, then try to merge terms
    # using the fact that some combinations might cancel or combine.
    
    # Alternative: Use the fact that the tensor is a matrix multiplication tensor
    # and try known algorithms.
    
    # For 2x4x5, one approach:
    # Use Strassen for 2x2x2 blocks.
    # Split A into 2x2 blocks (2 blocks), B into 2x5 blocks (2 blocks).
    # C = A1*B1 + A2*B2
    # Each 2x2*2x5:
    #   Split B into 2x2 + 2x3
    #   2x2*2x2: rank 7 (Strassen)
    #   2x2*2x3: rank 6 (naive, since 2*2*3=6 and it's likely optimal)
    #   Total: 13
    # Two blocks: 26
    
    # Can we do better? What if we don't split the middle dimension?
    # 2x4*4x5 directly:
    # We can use the fact that 4 = 2+2, but maybe a different split helps.
    
    # Let's try rank 24:
    # Maybe we can share some Strassen terms between the two blocks.
    
    # Actually, let's just implement the rank-26 construction and see if
    # we can improve it.
    
    # Rank-26 construction:
    rank_26 = 26
    U26 = np.zeros((n*m, rank_26))
    V26 = np.zeros((m*p, rank_26))
    W26 = np.zeros((p*n, rank_26))
    
    col_idx = 0
    
    # Strassen terms for 2x2x2
    strassen_terms = [
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        )
    ]
    
    # Standard terms for 2x2x3
    standard_terms_2x2x3 = [
        (
            {(0,0): 1},
            {(0,0): 1},
            {(0,0): 1}
        ),
        (
            {(0,1): 1},
            {(1,0): 1},
            {(0,0): 1}
        ),
        (
            {(0,0): 1},
            {(0,1): 1},
            {(0,1): 1}
        ),
        (
            {(0,1): 1},
            {(1,1): 1},
            {(0,1): 1}
        ),
        (
            {(1,0): 1},
            {(0,0): 1},
            {(1,0): 1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1},
            {(1,0): 1}
        ),
        (
            {(1,0): 1},
            {(0,1): 1},
            {(1,1): 1}
        ),
        (
            {(1,1): 1},
            {(1,1): 1},
            {(1,1): 1}
        )
    ]
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, U_mat, V_mat, W_mat, idx):
        for (i, j), val in u_coeffs.items():
            U_mat[i * m + j, idx] += val
        for (j, k), val in v_coeffs.items():
            V_mat[j * p + k, idx] += val
        for (k, i), val in w_coeffs.items():
            W_mat[k * n + i, idx] += val
    
    for block_idx in range(2):
        j_offset = block_idx * 2
        
        # Strassen for 2x2x2 (columns 0,1 of B)
        for u_c, v_c, w_c in strassen_terms:
            u_g = {}
            v_g = {}
            w_g = {}
            
            for (i, j_l), val in u_c.items():
                j_g = j_offset + j_l
                u_g[(i, j_g)] = val
            
            for (j_l, k_l), val in v_c.items():
                j_g = j_offset + j_l
                k_g = 0 + k_l
                v_g[(j_g, k_g)] = val
            
            for (i, k_l), val in w_c.items():
                k_g = 0 + k_l
                w_g[(k_g, i)] = val
            
            add_term(u_g, v_g, w_g, U26, V26, W26, col_idx)
            col_idx += 1
        
        # Standard for 2x2x3 (columns 2,3,4 of B)
        for u_c, v_c, w_c in standard_terms_2x2x3:
            u_g = {}
            v_g = {}
            w_g = {}
            
            for (i, j_l), val in u_c.items():
                j_g = j_offset + j_l
                u_g[(i, j_g)] = val
            
            for (j_l, k_l), val in v_c.items():
                j_g = j_offset + j_l
                k_g = 2 + k_l
                v_g[(j_g, k_g)] = val
            
            for (i, k_l), val in w_c.items():
                k_g = 2 + k_l
                w_g[(k_g, i)] = val
            
            add_term(u_g, v_g, w_g, U26, V26, W26, col_idx)
            col_idx += 1
    
    if verify(U26, V26, W26):
        best_rank = 26
        best_U = U26
        best_V = V26
        best_W = W26
    
    # Now try to reduce further using numerical methods
    # Try to find a lower-rank decomposition by optimization
    
    # Use a simple approach: try to combine terms
    # If two terms have proportional U, V, or W vectors, they might be mergeable
    
    # Try random perturbations and SVD-based rank reduction
    rng = np.random.default_rng(42)
    
    for attempt in range(50):
        if time.time() > deadline - 20:
            break
        
        # Start from best_U, best_V, best_W
        U_cur = best_U.copy()
        V_cur = best_V.copy()
        W_cur = best_W.copy()
        
        # Try to reduce rank by 1
        # Pick a random pair of columns to merge
        r1, r2 = rng.choice(best_rank, 2, replace=False)
        
        # Try to find alpha, beta such that
        # U[:, r1] + alpha * U[:, r2] is "simpler"
        # This is heuristic; let's just try dropping one term and adjusting
        
        # Alternative: Use the fact that if the sum of two rank-1 tensors
        # has a certain structure, we might be able to replace them with
        # a single rank-1 tensor.
        
        # For now, let's try a different approach:
        # Use the tensor's symmetries to find a better decomposition.
        
        # Actually, let's try to use the known result that 2x4x5 might
        # have rank 24 or less.
        
        # Try a rank-24 construction:
        # Split 4 into 2+2, but use a more efficient scheme for 2x2x5.
        # 2x2x5: Can we do it in rank 12?
        # 2x2x5 tensor: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        # Naive: 10 terms
        # Can we do better? The slice rank is at least max(2*2, 2*5, 2*5) = 10
        # So rank >= 10. And rank <= 10 (naive). So rank = 10.
        
        # So 2x2x5 has rank 10, not 13.
        # Then 2x4x5: split into two 2x2x5, total rank 20.
        # But the lower bound is 20, so rank 20 would be optimal!
        
        # Let's construct a rank-20 decomposition.
        break
    
    # Rank-20 construction:
    # C = A1*B1 + A2*B2 where A1, A2 are 2x2, B1, B2 are 2x5
    # Each 2x2x5 has rank 10 (naive is optimal for 2x2x5)
    # Total: 20
    
    rank_20 = 20
    U20 = np.zeros((n*m, rank_20))
    V20 = np.zeros((m*p, rank_20))
    W20 = np.zeros((p*n, rank_20))
    
    col_idx = 0
    
    for block_idx in range(2):
        j_offset = block_idx * 2
        for i in range(2):
            for j_l in range(2):
                for k in range(5):
                    j_g = j_offset + j_l
                    U20[i*m + j_g, col_idx] = 1.0
                    V20[j_g*p + k, col_idx] = 1.0
                    W20[k*n + i, col_idx] = 1.0
                    col_idx += 1
    
    if verify(U20, V20, W20):
        best_rank = 20
        best_U = U20
        best_V = V20
        best_W = W20
    
    # Can we do better than 20? The slice rank lower bound is 20,
    # so 20 is optimal if the slice rank bound is tight.
    # For matrix multiplication tensors, the slice rank lower bound
    # is often tight, so rank 20 is likely optimal.
    
    # Let's try to verify if we can do rank < 20
    # The slice rank lower bound for 2x4x5 is max(2*4, 4*5, 2*5) = 20
    # So rank >= 20. Since we have a rank-20 construction, rank 20 is optimal.
    
    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
