import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _build_schoolbook():
    """Build the schoolbook rank-75 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_strassen_block():
    """
    Build a Strassen-like decomposition for 2x2 by 2x2 matrix multiplication,
    then embed it into the 3x5 by 5x5 context.
    
    Strassen's algorithm computes 2x2 * 2x2 using 7 multiplications instead of 8.
    We'll build a rank-7 decomposition for the 2x2x2 tensor and then extend it.
    
    Actually, let's think more carefully. The matrix multiplication tensor for
    m x n * n x p has rank that is not simply the product of ranks of smaller
    blocks. Let me instead try a different approach.
    
    I'll construct a decomposition that uses the Strassen idea on a subset of
    the indices, reducing the effective rank.
    """
    n, m, p = DIMENSIONS
    # Start with schoolbook and then try to merge columns
    U, V, W = _build_schoolbook()
    rank = n * m * p
    
    # Try to find linear dependencies and merge columns
    # For each column c, check if it can be expressed as a combination of others
    # such that the tensor equality is preserved
    
    # A column c corresponds to a triple (i_c, j_c, k_c) where
    # U[i_c*m + j_c, c] = 1, V[j_c*p + k_c, c] = 1, W[k_c*n + i_c, c] = 1
    # (for the schoolbook case)
    
    # Let's try a more systematic approach: use the fact that for matrix
    # multiplication, certain tensor entries are related.
    
    # Actually, let me try a completely different strategy: use the known
    # lower bounds and try random projections to find a lower rank.
    
    return U, V, W

def _try_reduce_rank(U, V, W, max_iterations=100, deadline=None):
    """
    Attempt to reduce the rank of the decomposition by merging or eliminating
    columns that are linearly dependent in a way that preserves the tensor.
    
    The key insight: if column c1 = alpha * c2 + beta * c3 in all three matrices
    simultaneously (i.e., U[:,c1] = alpha*U[:,c2] + beta*U[:,c3], etc.), then
    we can eliminate c1 and adjust the coefficients.
    
    More generally, we look for columns that can be eliminated by finding
    a linear combination that reconstructs the same tensor contribution.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    if deadline is not None:
        start = time.time()
        time_limit = deadline - start - 5  # safety margin
    
    # Precompute the target tensor
    i_range = np.arange(n)
    j_range = np.arange(m)
    k_range = np.arange(p)
    I, J, K = np.meshgrid(i_range, j_range, k_range, indexing='ij')
    U_idx = I * m + J
    V_idx = J * p + K
    W_idx = K * n + I
    
    U_exp = U[U_idx, :]
    V_exp = V[V_idx, :]
    W_exp = W[W_idx, :]
    T_target = np.sum(U_exp * V_exp * W_exp, axis=-1)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to eliminate columns one by one
    for _ in range(max_iterations):
        if deadline is not None:
            if time.time() > time_limit:
                break
        
        progress = False
        for c in range(rank):
            if deadline is not None:
                if time.time() > time_limit:
                    break
            
            if rank <= 1:
                break
                
            # Try to eliminate column c by expressing it as a combination
            # of other columns
            # We need to find coefficients such that the tensor contribution
            # of column c can be replicated by adjusting other columns
            
            # Simple approach: check if removing column c and adjusting
            # others can still produce T_target
            
            U_no_c = np.delete(best_U, c, axis=1)
            V_no_c = np.delete(best_V, c, axis=1)
            W_no_c = np.delete(best_W, c, axis=1)
            new_rank = rank - 1
            
            # Check if the remaining columns can still span the tensor
            # This is a complex feasibility problem. For now, just check
            # if the column is "redundant" in a simple sense.
            
            # A column is redundant if there exists a linear combination of
            # the other columns that produces the same (U, V, W) triple.
            
            # Let's try: can we find a vector alpha such that
            # U[:,c] = U_no_c @ alpha
            # V[:,c] = V_no_c @ alpha  
            # W[:,c] = W_no_c @ alpha
            # simultaneously?
            
            # This is an overdetermined system. Let's try to solve it.
            # Stack the three equations
            # [U_no_c; V_no_c; W_no_c] @ alpha = [U[:,c]; V[:,c]; W[:,c]]
            
            A_stack = np.vstack([U_no_c, V_no_c, W_no_c])  # shape: (nm + mp + pn, new_rank)
            b_stack = np.concatenate([best_U[:, c], best_V[:, c], best_W[:, c]])
            
            # Solve least squares
            try:
                alpha, residuals, rank_s, sv = np.linalg.lstsq(A_stack, b_stack, rcond=None)
                
                # Check if the solution is exact (residuals ~ 0)
                if np.allclose(A_stack @ alpha, b_stack, atol=1e-9):
                    # Column c is exactly representable by other columns
                    # Remove it
                    best_U = U_no_c
                    best_V = V_no_c
                    best_W = W_no_c
                    rank = new_rank
                    best_rank = new_rank
                    progress = True
                    break
            except:
                pass
        
        if not progress:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix 
    by a 5x5 matrix.
    
    Strategy: Start with schoolbook rank-75, then iteratively reduce rank by
    eliminating linearly dependent columns.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = DIMENSIONS
    
    # Step 1: Build schoolbook decomposition
    U, V, W = _build_schoolbook()
    rank = n * m * p
    
    # Step 2: Try to reduce rank
    U, V, W, rank = _try_reduce_rank(U, V, W, max_iterations=50, deadline=deadline)
    
    # Verify the decomposition
    i_range = np.arange(n)
    j_range = np.arange(m)
    k_range = np.arange(p)
    I, J, K = np.meshgrid(i_range, j_range, k_range, indexing='ij')
    U_idx = I * m + J
    V_idx = J * p + K
    W_idx = K * n + I
    
    U_exp = U[U_idx, :]
    V_exp = V[V_idx, :]
    W_exp = W[W_idx, :]
    
    T_computed = np.sum(U_exp * V_exp * W_exp, axis=-1)
    T_expected = np.ones((n, m, p))
    
    if not np.allclose(T_computed, T_expected):
        # Fallback to schoolbook if verification fails
        U, V, W = _build_schoolbook()
        rank = n * m * p
    
    # Check time
    if time.time() > deadline - 10:
        # Return quickly
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
