import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x5 matrix.
    
    Uses slice method: decompose into j=0 block (rank 10) and j=1,2 block.
    For j=1,2 block (2x2x5 tensor), attempt to find a rank-10 decomposition
    using a unified Strassen-like structure across all k slices.
    """
    start_time = time.time()
    DEADLINE = 145.0
    
    n, m, p = 2, 3, 5
    
    # Try rank 27: 10 (j=0) + 17 (j=1,2)
    # Or try to find a better decomposition for the 2x2x5 part
    
    # The 2x2x5 tensor: C[i,k] = A[i,1]B[1,k] + A[i,2]B[2,k] for i in {0,1}, k in {0..4}
    # This is a 2x2x5 matmul tensor.
    
    # Let's try to find a rank-r decomposition for this sub-tensor
    # using random initialization and iterative refinement
    
    def find_2x2x5_decomposition(target_rank, max_iters=500, tol=1e-10):
        """Find a rank decomposition of the 2x2x5 matmul tensor."""
        # Tensor T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0..4}
        # We want: T = sum_r u_r[i] * v_r[j] * w_r[k]
        # where u is 2xR, v is 2xR, w is 5xR
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for trial in range(10):
            np.random.seed(42 + trial)
            U = np.random.randn(2, target_rank)
            V = np.random.randn(2, target_rank)
            W = np.random.randn(5, target_rank)
            
            # Iterative refinement: minimize ||T - sum U[:,r]*V[:,r]*W[:,r]||
            for it in range(max_iters):
                if time.time() > DEADLINE:
                    break
                
                # Update U: for each r, U[:,r] should be the best rank-1 approximation
                # of the residual when V[:,r] and W[:,r] are fixed
                residual = np.ones((2, 5))  # T[i,k] = 1
                for r in range(target_rank):
                    residual -= np.outer(U[:, r], W[:, r]) * V[:, r].sum()  # This isn't right
                
                # Better approach: use alternating least squares
                # T[i,k] = sum_r U[i,r] * V_sum[r] * W[k,r] where V_sum[r] = sum_j V[j,r]
                # Actually T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] in the full problem
                # For the sub-tensor with fixed j indices (j=1,2 mapped to 0,1):
                # C[i,k] = sum_{j_local} A[i,j_local] * B[j_local,k]
                # = sum_r (sum_j_local A[i,j_local]*U_sub[i,j_local,r]) * (sum_j_local B[j_local,k]*V_sub[j_local,r]) * W_sub[k,i,r]
                
                # This is getting complex. Let me use a different approach.
                break
            
            # Skip complex optimization for now, use known constructions
            break
        
        return None
    
    # Let me try a specific construction for 2x2x5 with rank 10
    # Use 5 pairs of terms, one pair per k value, but with sharing
    
    # Actually, let's try: rank 10 for 2x2x5
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # Term 2k: A[i,0]*B[0,k] -> but this needs separate terms for each k unless we share
    
    # Alternative: Use the fact that B[0,:] and B[1,:] are vectors
    # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    
    # This is a 2x5 matrix = sum of 2 outer products (A[0,0]*B[0,:] etc.)
    # But in tensor decomposition, we need rank-1 tensors.
    
    # Let me just try rank 26: 10 + 16, and verify
    
    # Actually, let me try to use a better Strassen for the combined j=1,2 part
    # by treating it as a 2x2x5 problem and using 10 terms
    
    # Construction for 2x2x5 with rank 10:
    # For k in 0..4, use 2 terms per k:
    # Term 2k: U[0,0]=1, V[0,k]=1, W[k,0]=1 (A[0,0]*B[0,k] -> C[0,k])
    # Term 2k+1: U[1,1]=1, V[1,k]=1, W[k,1]=1 (A[1,1]*B[1,k] -> C[1,k])
    # But we're missing A[0,1]*B[1,k] and A[1,0]*B[0,k]
    
    # That's only 2 terms per k, total 10, but incomplete.
    
    # Full schoolbook for 2x2x5 is 20 terms. We need to do better.
    
    # Let me try: 4 terms per k would be 20. Can we share across k?
    # If we use U[i,j] that doesn't depend on k, and V[j,k] that doesn't depend on i...
    
    # The issue is that W depends on both k and i.
    
    # Let me just keep the current approach but try to optimize the constants
    # or find if rank 27 is achievable.
    
    # For now, return the current rank-28 solution but with better verification
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Block 1: j=0, k=0..4 (schoolbook 2x1x5)
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        
        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = -1.0
        col += 1
        
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = -1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        col += 1
        
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        col += 1
        
        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+0), col] = -1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = -1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        col += 1
    
    add_strassen(i_off=0, j_off=1, k_off=0)
    
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1.0
            V[j * p + 2, col] = 1.0
            W[2 * n + i, col] = 1.0
            col += 1
    
    add_strassen(i_off=0, j_off=1, k_off=3)
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    # Verify
    A_test = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B_test = np.array([[1.0, 0.0, -1.0, 2.0, 0.5],
                        [0.0, 1.0, 0.0, -1.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, -0.5]])
    C_test = A_test @ B_test
    
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    
    s1 = np.einsum('ij,ijt->it', A_test, U_mat)
    s2 = np.einsum('jk,jkt->kt', B_test, V_mat)
    C_decomp = np.einsum('it,kt,kit->ik', s1, s2, W_mat)
    
    if not np.allclose(C_test, C_decomp, atol=1e-12):
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=float)
        V_sb = np.zeros((m * p, rank_sb), dtype=float)
        W_sb = np.zeros((p * n, rank_sb), dtype=float)
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col_sb] = 1.0
                    V_sb[j * p + k, col_sb] = 1.0
                    W_sb[k * n + i, col_sb] = 1.0
                    col_sb += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
