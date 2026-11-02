"""Rewritten program for 3x5x5 matrix multiplication rank minimization."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    
    Strategy: Use the known fact that the 5x5x5 multiplication tensor has rank 19.
    For 3x5x5, we can view this as multiplying a 3x5 matrix by a 5x5 matrix.
    The output is 3x5. We can decompose this using the structure of the 5x5x5 result.
    
    Key insight: The 3x5x5 multiplication can be seen as 3 copies of 1x5x5 multiplications,
    but with shared structure. However, a better approach is to note that:
    
    For C = A * B where A is 3x5 and B is 5x5:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can use the rank-19 decomposition of 5x5x5 multiplication. The standard result
    (Landsberg, 1997; Strassen, 1969 for smaller cases) shows that 5x5x5 has rank 19.
    
    For 3x5x5, one approach: treat the 3 rows of A as separate, giving 3 * rank(1x5x5) 
    but 1x5x5 is just a linear map, rank 5. So naive is 3*5=15? No, that's not right.
    
    Actually, let's think more carefully. The tensor for 3x5x5 multiplication has 
    dimension 3*5*5 = 75. The schoolbook rank is 3*5*5 = 75.
    
    A better decomposition: We can use the fact that for each column k of the output,
    C[:,k] = A * B[:,k], which is a 3x5 times 5x1 multiplication. This is a 3x5x1 
    multiplication, which has rank 3*5*1 = 15 in schoolbook, but can be done in rank 3 
    since it's just matrix-vector multiplication (each output is a linear combination).
    
    Wait, matrix-vector multiplication 3x5 * 5x1: the result is 3 values, each is a 
    dot product of a row of A with the vector. So we need 3*5 = 15 bilinear terms? No,
    each output C[i,k] = sum_j A[i,j] B[j,k]. For fixed k, we have 3 outputs, each 
    requiring 5 products. So 15 products per column, 5 columns = 75. That's schoolbook.
    
    To do better, we need to exploit structure. Let me use the known rank-19 decomposition
    for 5x5x5 and extend it.
    
    Actually, the correct approach for 3x5x5: 
    - The minimum rank is at most 3 * 5 = 15 if we can share the B factors.
    - But actually, each C[i,k] depends on all A[i,j] and B[j,k].
    
    Let me try a different approach: use the decomposition where we factor through 
    the 5x5 space. The tensor T_{i,j,k} = A_{i,j} * B_{j,k} * C_{i,k}.
    
    For rank r, we need:
    T_{i,j,k} = sum_c U_{i,j,c} * V_{j,k,c} * W_{i,k,c}
    
    where U has shape (3*5, r), V has shape (5*5, r), W has shape (5*3, r).
    
    Note the W indexing: W[k*n+i, c] per the convention, so W is indexed by (k,i) pairs.
    
    One known approach: use the fact that 5x5x5 multiplication can be done in 19 products.
    For 3x5x5, we can view it as 3 independent 1x5x5 problems, but sharing the B matrix.
    
    Actually, here's a cleaner approach: 
    C = A * B, where A is 3x5, B is 5x5.
    Write A = [A_1; A_2; A_3] where A_i is 1x5.
    Then C_i = A_i * B, which is 1x5 times 5x5 = 1x5.
    
    Each C_i = A_i * B can be computed with rank 19 (using the 5x5x5 result, but since 
    it's 1x5x5, actually it's just a matrix-vector product which has rank 5 in the 
    bilinear sense... wait no).
    
    For 1x5x5: C_k = sum_j A_j B_{j,k}. This is 5 products per output, 5 outputs = 25 
    schoolbook. But can we do better? The tensor is essentially the identity map from 
    R^5 x R^{5x5} -> R^5, which is linear in each factor. The rank of this tensor...
    
    Actually, for 1xm x mxn multiplication, the rank is m*n (you need each product A_j * B_{j,k}).
    So 1x5x5 has rank 25.
    
    So 3x5x5 naive is 3*25 = 75. To do better, we need to share products between the 
    three rows of A.
    
    Let me try a random search for a lower-rank decomposition, starting from schoolbook 
    and trying to reduce rank by finding linear dependencies.
    """
    start_time = time.time()
    deadline = start_time + 145  # Leave 5 seconds margin
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p  # 75
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Try to find a lower-rank decomposition using iterative compression
    # Strategy: Start with schoolbook, then try to find linear combinations that
    # allow us to eliminate some columns.
    
    # Method: Randomly perturb and use SVD-like approach to find low-rank approximations
    # that are exact.
    
    # Alternative: Use the known structure. For 3x5x5, one can use:
    # - Decompose B into 5 column vectors
    # - For each column, compute A * B_col (3 outputs)
    # - This gives 3*5 = 15 outputs, each requiring 5 products = 75 total
    
    # But we can share: if we precompute certain combinations of A's rows...
    
    # Let's try a systematic approach: assume rank r, and try to solve for U, V, W
    # such that the tensor identity holds.
    
    # The tensor identity: for all i,j,k:
    # 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is a system of nonlinear equations. We can use iterative methods.
    
    # Try ranks from a reasonable lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # Also, the tensor rank is at least the matrix rank of any unfolding.
    
    # Let's try to find a decomposition with rank < 75 using random initialization
    # and gradient-free optimization (e.g., alternating minimization).
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the 3x5x5 multiplication tensor."""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        # Check: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
        # And for all other combinations, it should be 0.
        # Actually, the tensor is: T[i,j,k] = 1 if the multiplication is valid, else 0.
        # For matrix mult, T[i,j,k] = 1 for all valid (i,j,k) in the multiplication.
        # The condition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        # Wait, that's not right. The tensor for matrix multiplication C = AB is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T has T[i,j,k] = 1 for the "multiplication" operation.
        # The decomposition means: delta[i,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # summed over j? No.
        
        # Let me re-read the convention. The standard tensor decomposition for matrix 
        # multiplication A (n x m) * B (m x p) -> C (n x p):
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T is in R^{(n*m) x (m*p) x (n*p)} where:
        # T[i*m+j, j*p+k, i*n+k] = 1 (using 0-indexing, but with the W convention W[k*n+i])
        
        # So the decomposition is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # where a = i*m+j, b = j'*p+k, c = k'*n+i'
        # and T[a,b,c] = 1 if a corresponds to (i,j), b to (j,k), c to (k,i) with matching
        # indices, i.e., the j in a equals the j' in b, and the k in b equals the k' in c,
        # and the i in a equals the i' in c.
        
        # More precisely: T[a,b,c] = 1 iff:
        #   a = i*m + j, b = j*p + k, c = k*n + i for some valid i,j,k
        #   (i.e., the j from a matches the j from b, k from b matches k from c, i from c matches i from a)
        
        # So we need: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b,c
        
        # Build the target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        
        # Compute the decomposition
        # For each r, the outer product U[:,r] * V[:,r] * W[:,r]
        # Sum over r
        # This can be done efficiently:
        # result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # Use einsum or manual
        # For small sizes, we can do it directly
        result = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            result += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        
        max_err = np.max(np.abs(result - T))
        return max_err < tol, max_err
    
    # First verify schoolbook
    ok, err = verify(best_U, best_V, best_W)
    if not ok:
        raise ValueError(f"Schoolbook verification failed: err={err}")
    
    # Try to reduce rank using alternating minimization
    # Start with a random rank-r initialization and try to fit the tensor
    
    def try_rank(r, max_iters=500, tol=1e-12):
        """Try to find a rank-r decomposition using alternating minimization."""
        n, m, p = DIMENSIONS
        
        # Initialize randomly
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Build target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        
        for it in range(max_iters):
            # Update U: for each a, U[a,:] should minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
            # This is a linear least squares problem.
            # Fix V, W. For each a:
            #   T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r]) outer product
            #   So T[a,:,:] = sum_r U[a,r] * M_r where M_r = V[:,r] * W[:,r] (outer product, shape (m*p, p*n))
            #   Stack: T[a,:] (shape (m*p*p*n,)) = U[a,:] (shape (r,)) @ M (shape (r, m*p*p*n))
            
            # Build M: M[r, b, c] = V[b,r] * W[c,r]
            M = np.einsum('br,cr->bcr', V, W)  # shape (m*p, p*n, r)
            M_flat = M.transpose(2, 0, 1).reshape(r, m*p*p*n)  # shape (r, m*p*p*n)
            
            # T_flat: shape (n*m, m*p*p*n)
            T_flat = T.reshape(n*m, m*p*p*n)
            
            # Solve: T_flat = U @ M_flat
            # U = T_flat @ M_flat^T @ (M_flat @ M_flat^T)^{-1}
            try:
                MMT = M_flat @ M_flat.T  # (r, r)
                MT_T = M_flat.T @ T_flat.T  # (r, n*m)
                U_new = np.linalg.solve(MMT, MT_T).T  # (n*m, r)
            except np.linalg.LinAlgError:
                U_new = np.linalg.lstsq(M_flat, T_flat.T, rcond=None)[0].T
            
            # Update V
            M2 = np.einsum('ar,cr->acr', U, W)  # shape (n*m, p*n, r)
            M2_flat = M2.transpose(2, 0, 1).reshape(r, n*m*p*n)
            T_flat2 = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
            try:
                MMT2 = M2_flat @ M2_flat.T
                MT_T2 = M2_flat.T @ T_flat2.T
                V_new = np.linalg.solve(MMT2, MT_T2).T
            except np.linalg.LinAlgError:
                V_new = np.linalg.lstsq(M2_flat, T_flat2.T, rcond=None)[0].T
            
            # Update W
            M3 = np.einsum('ar,br->abr', U, V)  # shape (n*m, m*p, r)
            M3_flat = M3.transpose(2, 0, 1).reshape(r, n*m*m*p)
            T_flat3 = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
            try:
                MMT3 = M3_flat @ M3_flat.T
                MT_T3 = M3_flat.T @ T_flat3.T
                W_new = np.linalg.solve(MMT3, MT_T3).T
            except np.linalg.LinAlgError:
                W_new = np.linalg.lstsq(M3_flat, T_flat3.T, rcond=None)[0].T
            
            U, V, W = U_new, V_new, W_new
        
        # Check error
        result = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            result += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        err = np.max(np.abs(result - T))
        return U, V, W, err
    
    # Try various ranks
    # Lower bound is at least max(n*m, m*p, p*n) = 25
    # But the true rank might be higher. Let's try from 25 upward.
    
    # Actually, let's be smarter. Try a few random initializations for each rank.
    
    for r in range(25, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            U, V, W, err = try_rank(r, max_iters=200)
            
            if err < 1e-10:
                # Found a valid decomposition!
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            # Try to go lower
            continue
        else:
            # If we can't find rank r, it's likely infeasible, but let's not assume
            # Try a few more trials
            for trial in range(5):
                if time.time() > deadline - 10:
                    break
                U, V, W, err = try_rank(r, max_iters=300)
                if err < 1e-10:
                    best_rank = r
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
    
    # Final verification
    ok, err = verify(best_U, best_V, best_W)
    if not ok:
        # Fall back to schoolbook
        best_rank = rank_schoolbook
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
    
    # Convert to lists of lists with integer/half-integer entries if possible
    # The entries might be floating point; round to nearest half-integer if close
    def to_half_integer(x):
        """Round to nearest half-integer if close, else keep as float."""
        hx = np.round(x * 2) / 2
        if np.allclose(x, hx, atol=1e-10):
            return hx
        return x
    
    best_U = to_half_integer(best_U)
    best_V = to_half_integer(best_V)
    best_W = to_half_integer(best_W)
    
    # Verify again after rounding
    ok, err = verify(best_U, best_V, best_W)
    if not ok:
        # Don't round, use original
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
        best_rank = rank_schoolbook
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
