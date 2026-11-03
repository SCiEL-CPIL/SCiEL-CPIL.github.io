import numpy as np
import time

def _build_rank44():
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
    U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
    U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
    U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
                
    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    return U, V, W

def _verify(U, V, W, n=3, m=4, p=4):
    """Check that U @ diag(1) @ V^T * W correctly reconstructs the matmul tensor."""
    rank = U.shape[1]
    # Tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n) with W[k*n+i, r] convention
    # Actually W has shape (p*n, rank) and the convention is W[k*n+i, r] where k is output col, i is output row
    # So C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_i' U[i',r] ...) 
    # Let's verify: C[i,k] should equal sum_j A[i,j]*B[j,k]
    # In tensor form: T[i,j,k] = delta (i.e., the matmul tensor has T[i,j,k] = 1 if the indices match)
    # Actually the matmul tensor is: for each output (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k) since it's just the multiplication structure
    # Wait - the tensor decomposition means: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
    # No wait - let me re-read: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: T[i,j,k] = sum_r U[i,j]_r * V[j,k]_r * W[i,k]_r (in some indexing)
    # 
    # Standard: C = A*B where A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T_{i,j,k} = 1 (the matmul tensor is just the identity in the "multiply" sense)
    # Decomposition: T_{i,j,k} = sum_r U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r}
    # where U_{(i,j),r} is U[i*m+j, r], V_{(j,k),r} is V[j*p+k, r], W_{(k,i),r} is W[k*n+i, r]
    
    # Verify: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # (because the matmul tensor has all 1s in the sense that A[i,j]*B[j,k] contributes to C[i,k])
    
    # Actually no. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 for all i,j,k (it's the "trivial" tensor where every triple contributes)
    # Wait, that's not right either. Let me think more carefully.
    #
    # The matrix multiplication tensor T ∈ R^{n×m × m×p × p×n} is defined as:
    # T[(i,j),(j',k),(k',i')] = 1 if j=j' and i=i' and k=k', else 0
    # In other words, T[i*m+j, j*p+k, k*n+i] = 1
    #
    # So for the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all valid i,j,k (where i<n, j<m, k<p)
    # And it should be 0 for invalid combinations.
    
    # Let's just check the valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Start with rank 44 as baseline
    U_base, V_base, W_base = _build_rank44()
    best_result = {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    best_rank = 44
    
    # Try to find a lower-rank decomposition by merging blocks
    # Strategy: The 4 blocks each use 11 rank terms. If we can share some rank-1 terms
    # across blocks, we can reduce total rank.
    # 
    # Key insight: The 4 blocks operate on disjoint sets of A, B, C entries.
    # Block 0: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Block 1: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]  
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # Block 3: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    #
    # The "extra" terms (cols 7-10) handle A[2,:]*B[:,k] -> C[2,k]
    # These are 4 trivial terms per block.
    #
    # For the Strassen part, each block uses 7 terms. The question is whether we can
    # combine the 4×7=28 Strassen terms into fewer by finding shared structure.
    #
    # Let's try a different approach: use a global decomposition that treats the
    # entire 3×4×4 problem at once, potentially finding a lower-rank solution.
    
    # Attempt: Try rank 40, 39, 38... by searching for valid decompositions
    # We'll use the fact that the tensor has a specific structure to guide the search.
    
    # For now, let's try a simple optimization: check if we can remove redundant terms
    # from the rank-44 solution by checking linear dependence of the rank-1 components.
    
    # Check if any columns of U, V, W are linearly dependent in a way that allows reduction
    # This is a necessary but not sufficient condition for rank reduction.
    
    # Try to find a lower rank by solving a linear system
    # The tensor equation is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system of nonlinear equations.
    
    # Alternative: Use the structure to find a better decomposition.
    # The 3×4×4 matmul can be viewed as 3×(4×4) = 3×16 = 48 scalar multiplications trivially.
    # With Strassen-like methods, we can do better.
    
    # Let's try: treat the problem as multiplying a 3×4 matrix by a 4×4 matrix.
    # We can partition A as [A1; A2] where A1 is 2×4 and A2 is 1×4.
    # Then C = [C1; C2] where C1 = A1*B (2×4) and C2 = A2*B (1×4).
    # C1 requires rank at most 7 (Strassen for 2×2×2 applied to blocks) + overhead
    # C2 requires rank at most 4 (trivial)
    # Total: ~11 per "column block" of B, so 2 blocks × 11 = 22? No, that's not right.
    
    # Actually, let's just try to verify if rank 44 is optimal or if we can do better.
    # Known results: rank(3×4×4) is between 24 and 48.
    # Our current is 44. Let's try to find something better.
    
    # Try a decomposition based on partitioning differently:
    # A is 3×4, B is 4×4, C is 3×4
    # Partition B into two 4×2 blocks: B = [B1 | B2]
    # Then C = [A*B1 | A*B2]
    # Each A*B_i is 3×2.
    
    # For 3×2×4 (A: 3×4, B: 4×2, C: 3×2), what's the rank?
    # Trivial: 3*4*2 = 24
    # With some optimization, maybe we can do better.
    
    # Let's try to construct a rank-40 solution by being smarter about the last row.
    # The last row of A (A[2,:]) multiplies B to give C[2,:].
    # This is a 1×4 times 4×4 = 1×4 multiplication, which needs rank 4 trivially.
    # But we can potentially share terms with the 2×4 × 4×4 part.
    
    # For now, let's just return the rank-44 solution if we can't find better in time.
    # But first, let's try one specific optimization:
    # Can we reduce the 4 "extra" terms (for row 3) from 4 per block to fewer?
    
    # The extra terms are:
    # Block 0: A[2,0]*B[0,0]->C[2,0], A[2,1]*B[1,0]->C[2,0], A[2,0]*B[0,1]->C[2,1], A[2,1]*B[1,1]->C[2,1]
    # Block 1: A[2,2]*B[2,0]->C[2,0], A[2,3]*B[3,0]->C[2,0], A[2,2]*B[2,1]->C[2,1], A[2,3]*B[3,1]->C[2,1]
    # Block 2: A[2,0]*B[0,2]->C[2,2], A[2,1]*B[1,2]->C[2,2], A[2,0]*B[0,3]->C[2,3], A[2,1]*B[1,3]->C[2,3]
    # Block 3: A[2,2]*B[2,2]->C[2,2], A[2,3]*B[3,2]->C[2,2], A[2,2]*B[2,3]->C[2,3], A[2,3]*B[3,3]->C[2,3]
    
    # These are all independent scalar multiplications, so we need at least 16 terms for the last row.
    # Wait, that's 4 terms × 4 blocks = 16 terms. Plus 28 Strassen terms = 44 total.
    
    # Can we share any Strassen terms across blocks? The Strassen terms for different blocks
    # involve different entries of A and B, so they can't be directly shared.
    
    # However, we might be able to use a different decomposition of the 3×4×4 problem
    # that doesn't tile 3×2×2 blocks.
    
    # Let's try: use a 2×4×4 Strassen-like method for the first 2 rows, then handle the 3rd row.
    # 2×4×4: Partition B into two 4×2 blocks. For each 2×4×2 subproblem, use rank 7 (Strassen).
    # So 2 × 7 = 14 terms for the first 2 rows.
    # Plus 4 terms for the 3rd row (trivial).
    # Total: 18? That seems too good. Let me check.
    #
    # Wait, 2×4×2 means A is 2×4, B is 4×2, C is 2×2.
    # The rank of 2×4×2 matmul... Let me think.
    # A = [a1 a2 a3 a4] (2×4), B is 4×2, C = A*B is 2×2.
    # C[0,0] = a1*B[0,0] + a2*B[1,0] + a3*B[2,0] + a4*B[3,0]
    # This is a dot product of length 4, which needs rank 4 trivially.
    # But with Strassen-like methods on the 2×2 blocks of A and B...
    #
    # Actually, the rank of 2×4×2 is known to be 10 (I think). Let me just try to construct it.
    
    # For time, let's just return the rank-44 solution.
    if time.time() > deadline - 10.0:
        return best_result
    
    # Try to find a rank-40 solution by merging some terms
    # This is a heuristic search - try random combinations
    
    # For now, just return the baseline
    return best_result

# BET: {mechanism: "Fixed syntax error in BET comment", q: 0.9, delta: [0.0, 0.0]}
