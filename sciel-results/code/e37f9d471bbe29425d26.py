"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check whether U,V,W exactly decompose the matmul tensor.
    Convention: W[k*n+i, c].
    Tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] must equal 1 for all (i,j,k).
    """
    rank = U.shape[1]
    # Build the tensor via einsum or explicit loops
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
    # Check all entries are exactly 1 (within tight tolerance for float)
    # For exact integer/half-integer, we can check with tolerance
    if not np.allclose(T, 1.0, atol=1e-9, rtol=1e-9):
        return False
    return True

def _try_rank(n, m, p, rank, max_iters=2000, seed=42):
    """Try to find a decomposition of given rank via iterative refinement.
    Returns (U, V, W) or None.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5

    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # We want: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Use gradient-free iterative method: for each factor, solve least squares
    # given the other two fixed.
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for it in range(max_iters):
        # Update W: for each (k,i), we need sum_c W[k*n+i,c] * (sum over j U[i*m+j,c]*V[j*p+k,c]) = 1
        # Let A_w[k,i,c] = sum_j U[i*m+j,c] * V[j*p+k,c]
        # Then W[k*n+i,:] * A_w[k,i,:]^T = 1, i.e., W_row dot A_w_row = 1
        # This is: for each (k,i), find w such that w @ A_w[k,i] = 1
        # A_w[k,i] is a vector of length rank: A_w[k,i,c] = sum_j U3[i,j,c]*V3[j,k,c]
        
        # Compute A_w: shape (p, n, rank)
        # A_w[k,i,c] = sum_j U3[i,j,c] * V3[j,k,c]
        # = einsum('ijc,jkc->kic', U3, V3)
        A_w = np.einsum('ijc,jkc->kic', U3, V3, optimize=True)
        
        # For each (k,i), solve w @ A_w[k,i] = 1
        # If A_w[k,i] is nonzero, w = A_w[k,i] / ||A_w[k,i]||^2
        # This gives the minimum-norm solution to w·a = 1.
        norms_sq = np.sum(A_w ** 2, axis=2)  # shape (p, n)
        # Avoid division by zero
        valid = norms_sq > 1e-12
        W3_new = np.zeros_like(W3)
        W3_new[valid] = A_w[valid] / norms_sq[valid, np.newaxis]
        # For invalid entries, keep old value or set to something
        W3_new[~valid] = W3[~valid]
        W3 = W3_new
        W = W3.reshape(p * n, rank)
        
        # Update U: for each (i,j), need sum_c U[i*m+j,c] * (sum over k V[j*p+k,c]*W[k*n+i,c]) = 1
        # A_u[i,j,c] = sum_k V3[j,k,c] * W3[k,i,c]
        A_u = np.einsum('jkc,kic->ijc', V3, W3, optimize=True)
        norms_sq_u = np.sum(A_u ** 2, axis=2)
        valid_u = norms_sq_u > 1e-12
        U3_new = np.zeros_like(U3)
        U3_new[valid_u] = A_u[valid_u] / norms_sq_u[valid_u, np.newaxis]
        U3_new[~valid_u] = U3[~valid_u]
        U3 = U3_new
        U = U3.reshape(n * m, rank)
        
        # Update V: for each (j,k), need sum_c V[j*p+k,c] * (sum over i U[i*m+j,c]*W[k*n+i,c]) = 1
        # A_v[j,k,c] = sum_i U3[i,j,c] * W3[k,i,c]
        A_v = np.einsum('ijc,kic->jkc', U3, W3, optimize=True)
        norms_sq_v = np.sum(A_v ** 2, axis=2)
        valid_v = norms_sq_v > 1e-12
        V3_new = np.zeros_like(V3)
        V3_new[valid_v] = A_v[valid_v] / norms_sq_v[valid_v, np.newaxis]
        V3_new[~valid_v] = V3[~valid_v]
        V3 = V3_new
        V = V3.reshape(m * p, rank)
        
        # Check convergence
        if it % 50 == 0 or it == max_iters - 1:
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
            err = np.max(np.abs(T - 1.0))
            if err < 1e-8:
                return U, V, W
    
    # Final check
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
    if np.max(np.abs(T - 1.0)) < 1e-8:
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with the known schoolbook solution
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But there may be better lower bounds. Let's try from a reasonable range.
    # Known: for 3x5x7, the tensor rank is at most 105. 
    # Let's try various ranks.
    
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound upward
    for rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random seeds
        found = False
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            result = _try_rank(n, m, p, rank, max_iters=500, seed=seed)
            if result is not None:
                U, V, W = result
                # Validate
                if _validate(U, V, W, n, m, p):
                    best_rank = rank
                    best = (U, V, W)
                    found = True
                    break
        
        if found:
            # We found a decomposition at this rank, no need to try higher ranks
            break
    
    # Convert to lists for the return format
    U_list = best[0].tolist()
    V_list = best[1].tolist()
    W_list = best[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
