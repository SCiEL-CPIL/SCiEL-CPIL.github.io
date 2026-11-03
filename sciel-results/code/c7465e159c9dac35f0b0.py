import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 5x5 matrix multiplication.
    Uses block decomposition 5 = 3 + 2 with known low-rank algorithms for sub-blocks.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Block split: 5 = 3 + 2
    # A is 5x5, split into A[0:3,0:3], A[0:3,3:5], A[3:5,0:3], A[3:5,3:5]
    # B is 5x5, split similarly
    # C is 5x5, split similarly
    #
    # C[0:3,0:3] = A[0:3,0:3]*B[0:3,0:3] + A[0:3,3:5]*B[3:5,0:3]
    # C[0:3,3:5] = A[0:3,0:3]*B[0:3,3:5] + A[0:3,3:5]*B[3:5,3:5]
    # C[3:5,0:3] = A[3:5,0:3]*B[0:3,0:3] + A[3:5,3:5]*B[3:5,0:3]
    # C[3:5,3:5] = A[3:5,0:3]*B[0:3,3:5] + A[3:5,3:5]*B[3:5,3:5]
    
    # Sub-problems:
    # 1. 3x3x3: A[0:3,0:3] * B[0:3,0:3]
    # 2. 3x2x3: A[0:3,3:5] * B[3:5,0:3]
    # 3. 3x3x2: A[0:3,0:3] * B[0:3,3:5]
    # 4. 3x2x2: A[0:3,3:5] * B[3:5,3:5]
    # 5. 2x3x3: A[3:5,0:3] * B[0:3,0:3]
    # 6. 2x2x3: A[3:5,3:5] * B[3:5,0:3]
    # 7. 2x3x2: A[3:5,0:3] * B[0:3,3:5]
    # 8. 2x2x2: A[3:5,3:5] * B[3:5,3:5]
    
    # For each sub-problem, we need a tensor decomposition.
    # The overall U, V, W matrices will be block-diagonal (in a sense) with
    # the sub-decompositions.
    
    # Let's build the decomposition piece by piece.
    # U has shape (n*m, rank) = (25, rank)
    # V has shape (m*p, rank) = (25, rank)
    # W has shape (p*n, rank) = (25, rank)
    #
    # W[k*n+i, r] corresponds to C[i,k] in the output.
    
    # For the schoolbook decomposition, each term (i,j,k) gives:
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    
    # For a block decomposition, we need to map the sub-problem indices
    # to the global indices.
    
    # Let's define helper functions for each sub-problem.
    
    def make_schoolbook(m, n, p):
        """Schoolbook decomposition for m x n x p matrix mult."""
        rank = m * n * p
        U = np.zeros((m * n, rank))
        V = np.zeros((n * p, rank))
        W = np.zeros((p * m, rank))
        c = 0
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    U[i * n + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * m + i, c] = 1
                    c += 1
        return U, V, W, rank
    
    def make_strassen_2x2():
        """Strassen's algorithm for 2x2x2, rank 7."""
        # Indices: A[i,j] with i,j in {0,1}, B[j,k] with j,k in {0,1}
        # C[i,k] = sum_j A[i,j] * B[j,k]
        #
        # U has shape (2*2, 7) = (4, 7)
        # V has shape (2*2, 7) = (4, 7)
        # W has shape (2*2, 7) = (4, 7)
        #
        # Strassen's 7 products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # U[i*2+j, c] for A[i,j]
        # V[j*2+k, c] for B[j,k]
        # W[k*2+i, c] for C[i,k]
        
        # M1 = (A00+A11)(B00+B11)
        # A00: U[0,0]=1, A11: U[3,0]=1
        # B00: V[0,0]=1, B11: V[3,0]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C00 += M1, C11 += M1
        W[0, 0] = 1; W[3, 0] = 1  # C[0,0] and C[1,1]
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C[1,0] += M2
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # C[0,1] += M3
        
        # M4 = A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1  # C[0,0] += M4, C[1,0] += M4
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[0, 4] = -1  # C[0,1] += M5, C[0,0] -= M5
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C[1,1] += M6
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # C[0,0] += M7
        
        return U, V, W, 7
    
    def make_3x3x3_rank23():
        """
        A known rank-23 decomposition for 3x3x3 matrix multiplication.
        This is complex; let me use a known construction.
        """
        # I'll use a known 23-term decomposition.
        # This is based on the work of Bini, Capovani, Lotti, Romani (1978)
        # or similar. Let me try to find an explicit form.
        
        # Actually, let me use a different approach: I'll try to find
        # the decomposition through a search or use a known formula.
        
        # For now, let me use schoolbook (rank 27) as a fallback
        # and try to improve.
        U, V, W, rank = make_schoolbook(3, 3, 3)
        return U, V, W, rank
    
    # Let me try a different approach: use a search to find a lower rank
    # decomposition for the full 5x5x5 problem.
    
    # Actually, let me just implement the block decomposition with
    # schoolbook for all sub-problems first, verify it works, then
    # try to improve.
    
    # Block decomposition:
    # Sub-problem (block_a_rows, block_a_cols, block_b_cols) -> output (block_a_rows, block_b_cols)
    # 
    # The global index mapping:
    # A is indexed by (i, j) where i is row, j is col. Global U index: i*5 + j
    # B is indexed by (j, k) where j is row, k is col. Global V index: j*5 + k
    # C is indexed by (i, k). Global W index: k*5 + i
    
    # Sub-problems with their local-to-global index mappings:
    # 1. A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3]
    #    Local A[i,j] -> Global U[i*5+j] for i,j in 0..2
    #    Local B[j,k] -> Global V[j*5+k] for j,k in 0..2
    #    Local C[i,k] -> Global W[k*5+i] for i,k in 0..2
    #
    # 2. A[0:3,3:5] * B[3:5,0:3] -> C[0:3,0:3]
    #    Local A[i,j] (j in 0..1) -> Global U[i*5+(3+j)] for i in 0..2, j in 0..1
    #    Local B[j,k] (j in 0..1) -> Global V[(3+j)*5+k] for j in 0..1, k in 0..2
    #    Local C[i,k] -> Global W[k*5+i] for i,k in 0..2
    #
    # etc.
    
    # Let me define the sub-problems as tuples:
    # (a_row_start, a_row_end, a_col_start, a_col_end, 
    #  b_row_start, b_row_end, b_col_start, b_col_end,
    #  c_row_start, c_row_end, c_col_start, c_col_end)
    
    sub_problems = [
        # (a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1)
        (0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3),  # 3x3x3
        (0, 3, 3, 5, 3, 5, 0, 3, 0, 3, 0, 3),  # 3x2x3
        (0, 3, 0, 3, 0, 3, 3, 5, 0, 3, 3, 5),  # 3x3x2
        (0, 3, 3, 5, 3, 5, 3, 5, 0, 3, 3, 5),  # 3x2x2
        (3, 5, 0, 3, 0, 3, 0, 3, 3, 5, 0, 3),  # 2x3x3
        (3, 5, 3, 5, 3, 5, 0, 3, 3, 5, 0, 3),  # 2x2x3
        (3, 5, 0, 3, 0, 3, 3, 5, 3, 5, 3, 5),  # 2x3x2
        (3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5),  # 2x2x2
    ]
    
    # For each sub-problem, get a decomposition and map to global indices.
    U_global = []
    V_global = []
    W_global = []
    
    for (a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1) in sub_problems:
        m = a_r1 - a_r0  # rows of A = rows of C
        nn = a_c1 - a_c0  # cols of A = rows of B
        pp = b_c1 - b_c0  # cols of B = cols of C
        
        # Get decomposition for this sub-problem
        if m == 2 and nn == 2 and pp == 2:
            U_sub, V_sub, W_sub, rank_sub = make_strassen_2x2()
        else:
            U_sub, V_sub, W_sub, rank_sub = make_schoolbook(m, nn, pp)
        
        # Map local indices to global
        U_mapped = np.zeros((25, rank_sub))
        V_mapped = np.zeros((25, rank_sub))
        W_mapped = np.zeros((25, rank_sub))
        
        for i_local in range(m):
            i_global = a_r0 + i_local
            for j_local in range(nn):
                j_global = a_c0 + j_local
                U_global_idx = i_global * 5 + j_global
                U_mapped[U_global_idx, :] = U_sub[i_local * nn + j_local, :]
        
        for j_local in range(nn):
            j_global = b_r0 + j_local
            for k_local in range(pp):
                k_global = b_c0 + k_local
                V_global_idx = j_global * 5 + k_global
                V_mapped[V_global_idx, :] = V_sub[j_local * pp + k_local, :]
        
        for k_local in range(pp):
            k_global = c_c0 + k_local
            for i_local in range(m):
                i_global = c_r0 + i_local
                W_global_idx = k_global * 5 + i_global
                W_mapped[W_global_idx, :] = W_sub[k_local * m + i_local, :]
        
        U_global.append(U_mapped)
        V_global.append(V_mapped)
        W_global.append(W_mapped)
    
    # Concatenate all sub-decompositions
    U_final = np.hstack(U_global)
    V_final = np.hstack(V_global)
    W_final = np.hstack(W_global)
    
    rank_final = U_final.shape[1]
    
    # Verify correctness
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition exactly computes matrix multiplication."""
        # The tensor T[i,j,k] = 1 should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, the convention is W[k*n+i, r], so W is indexed by (k, i) -> k*n+i
        # Let me re-read: "W[k*n+i,r]"
        # So W[row, col] where row = k*n + i, meaning W is indexed by output (i, k) as k*n+i
        # 
        # The decomposition should satisfy:
        # For all i, j, k: delta[j,j] ... no.
        # 
        # Matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor: T[i,j,k] = 1 (the multiplication tensor has 1s where the indices match)
        # Actually, the tensor for matrix mult is:
        # T[i, j, k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
        # More precisely, the tensor is T such that for any A, B:
        # C = A*B iff C[i,k] = sum_{j} A[i,j] B[j,k]
        # The tensor decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
        # where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[i,k]
        # Wait, I need to be careful with the index conventions.
        
        # Let me just check: for random A, B, does the decomposition give C = A*B?
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C using the decomposition
        # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
        # Wait, that's not right either. Let me think again.
        
        # The standard tensor decomposition for matrix mult:
        # C = A * B
        # C[i,k] = sum_j A[i,j] B[j,k]
        # 
        # Tensor form: C = sum_r (U_r^T A) ⊙ (B V_r) * W_r
        # where U_r is a vector of length n*m, V_r of length m*p, W_r of length p*n
        # and ⊙ is element-wise, and the reshaping is:
        # U_r reshaped to (n,m), V_r reshaped to (m,p), W_r reshaped to (p,n)
        # 
        # So: C = sum_r reshape(U_r, (n,m))^T * A ... no.
        
        # Let me use the direct formula:
        # C[i,k] = sum_r (sum_j A[i,j] U_r[i*m+j]) * (sum_j B[j,k] V_r[j*p+k]) * W_r[k*n+i]
        # 
        # Hmm, that doesn't look right. Let me think about it differently.
        
        # The tensor T for matrix mult has T[i,j,k] = 1 for all i,j,k (with the understanding
        # that the "multiplication" is the contraction over j).
        
        # Actually, the matrix multiplication tensor is defined as:
        # T[i, j, k] = 1 (for all i, j, k) -- no, that's not right.
        
        # Let me look at it from the decomposition perspective:
        # We want: for all A, B, C = A*B where C[i,k] = sum_j A[i,j] B[j,k]
        # 
        # If we have U, V, W with the given shapes, the decomposition means:
        # C[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r]) * W[k*n+i, r]
        # 
        # Wait, that would be: C[i,k] = sum_r W[k*n+i, r] * (A * U_r)[i] * (V_r^T * B)[k]
        # where (A * U_r)[i] = sum_j A[i,j] U_r[i*m+j] ... no, U_r is a vector, not a matrix.
        
        # I think the correct interpretation is:
        # Let U_r be reshaped to a matrix U_r_mat of shape (n, m)
        # Let V_r be reshaped to a matrix V_r_mat of shape (m, p)
        # Let W_r be reshaped to a vector w_r of shape (p*n) or matrix of shape (p, n)
        # 
        # Then: C = sum_r (A^T * U_r_mat) * (V_r_mat^T * B^T)^T * ... 
        # 
        # This is getting confusing. Let me just test empirically.
        
        # For the schoolbook decomposition:
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for term (i,j,k)
        # 
        # Then: C[i,k] = sum_c U[i*m+?, c] * V[?*p+k, c] * W[k*n+i, c]
        # For term (i',j',k'), U[i'*m+j', c] = 1, so it contributes to C[i',k'] only if i=i' and k=k'
        # and the contribution is V[j'*p+k, c] = 1 only if j'=j... 
        
        # I think the correct formula is:
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j and r
        # No, that's not right either.
        
        # Let me just try: C[i,k] = sum_j sum_r A[i,j] * B[j,k] * (U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r])
        # For schoolbook, U[i*m+j, r] = 1 when r corresponds to (i,j,k'), V[j*p+k, r] = 1 when r corresponds to (i',j,k), 
        # W[k*n+i, r] = 1 when r corresponds to (i',j',k). 
        # So the product is 1 only when i=i', j=j, k=k', giving C[i,k] = sum_j A[i,j]*B[j,k]. Correct!
        
        # So the formula is: C[i,k] = sum_j sum_r A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that's summing over j and r, but U, V, W are indexed by r. Let me re-examine.
        
        # Actually, I think it's:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r])
        # No, that would be a product of two sums, which is not what we want.
        
        # Let me look at the schoolbook case more carefully.
        # For schoolbook, each term r corresponds to a unique (i,j,k).
        # U[i*m+j, r] = 1 if r = (i,j,k) for some k, else 0.
        # V[j*p+k, r] = 1 if r = (i,j,k) for some i, else 0.
        # W[k*n+i, r] = 1 if r = (i,j,k) for some j, else 0.
        
        # So for a given r = (i_r, j_r, k_r):
        # U[i*m+j, r] = 1 iff i = i_r and j = j_r
        # V[j*p+k, r] = 1 iff j = j_r and k = k_r
        # W[k*n+i, r] = 1 iff k = k_r and i = i_r
        
        # So the product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 iff i=i_r, j=j_r, k=k_r.
        
        # Therefore: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        # But this sums over r, and for each r, we need specific i, j, k. 
        # 
        # I think the correct formula is:
        # C[i,k] = sum_j A[i,j] * B[j,k] * (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r])
        # 
        # For schoolbook, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (since exactly one r matches).
        # So C[i,k] = sum_j A[i,j] * B[j,k] * 1 = (A*B)[i,k]. Correct!
        
        # So the verification formula is:
        # C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k]
        # where T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And we need T[i,j,k] = 1 for all i,j,k.
        
        T = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            U_vec = U[:, r]  # length n*m
            V_vec = V[:, r]  # length m*p
            W_vec = W[:, r]  # length p*n
            U_mat = U_vec.reshape(n, m)
            V_mat = V_vec.reshape(m, p)
            W_mat = W_vec.reshape(p, n)
            # T[i,j,k] += U_mat[i,j] * V_mat[j,k] * W_mat[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U_mat[i,j] * V_mat[j,k] * W_mat[k,i]
        
        # Check if T is all 1s
        return np.allclose(T, 1.0, atol=1e-10)
    
    # Verify
    is_valid = verify(U_final, V_final, W_final, n, m, p)
    
    if not is_valid:
        # Fall back to schoolbook
        U_fb, V_fb, W_fb, rank_fb = make_schoolbook(n, m, p)
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
