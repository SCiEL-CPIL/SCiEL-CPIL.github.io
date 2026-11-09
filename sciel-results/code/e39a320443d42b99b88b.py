"""Rank-minimization via direct tensor optimization: start from rank-44 Strassen
decomposition and iteratively project out linear dependencies to reach a lower
target rank, minimizing the reconstruction error of the multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _multiplication_tensor(n, m, p):
    """Build the exact multiplication tensor T with T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _multiplication_tensor(n, m, p)
    rank = U.shape[1]
    result = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(result, T, atol=tol)

def _build_strassen_2x4x4():
    """
    Strassen decomposition for 2x4x4 matrix multiplication.
    Returns U (8, 28), V (16, 28), W (8, 28) where W uses index k*2+i.
    """
    rank = 28
    U = np.zeros((8, rank))
    V = np.zeros((16, rank))
    W = np.zeros((8, rank))

    col = 0
    for s in range(7):
        if s == 0:  # M1 = (A11+A22)(B11+B22)
            X = {0: [(0,0,1),(1,2,1)], 1: [(0,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1),(2,0,1)], (0,1): [(0,1,1),(2,1,1)],
                 (1,0): [(1,0,1),(3,0,1)], (1,1): [(1,1,1),(3,1,1)]}
            out = {0: [(0,0,1),(1,2,1)], 1: [(0,1,1),(1,3,1)]}
        elif s == 1:  # M2 = (A21+A22)B11
            X = {0: [(1,0,1),(1,2,1)], 1: [(1,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1)], (0,1): [(0,1,1)], (1,0): [(1,0,1)], (1,1): [(1,1,1)]}
            out = {0: [(1,0,1),(1,2,-1)], 1: [(1,1,1),(1,3,-1)]}
        elif s == 2:  # M3 = A11(B12-B22)
            X = {0: [(0,0,1)], 1: [(0,1,1)]}
            Y = {(0,0): [(0,2,1),(2,2,-1)], (0,1): [(0,3,1),(2,3,-1)],
                 (1,0): [(1,2,1),(3,2,-1)], (1,1): [(1,3,1),(3,3,-1)]}
            out = {0: [(0,2,1),(1,2,1)], 1: [(0,3,1),(1,3,1)]}
        elif s == 3:  # M4 = A22(B21-B11)
            X = {0: [(1,2,1)], 1: [(1,3,1)]}
            Y = {(0,0): [(2,0,1),(0,0,-1)], (0,1): [(2,1,1),(0,1,-1)],
                 (1,0): [(3,0,1),(1,0,-1)], (1,1): [(3,1,1),(1,1,-1)]}
            out = {0: [(0,0,1),(1,0,1)], 1: [(0,1,1),(1,1,1)]}
        elif s == 4:  # M5 = (A11+A12)B22
            X = {0: [(0,0,1),(0,2,1)], 1: [(0,1,1),(0,3,1)]}
            Y = {(0,0): [(2,2,1)], (0,1): [(2,3,1)], (1,0): [(3,2,1)], (1,1): [(3,3,1)]}
            out = {0: [(0,0,-1),(0,2,1)], 1: [(0,1,-1),(0,3,1)]}
        elif s == 5:  # M6 = (A21-A11)(B11+B12)
            X = {0: [(1,0,1),(0,0,-1)], 1: [(1,1,1),(0,1,-1)]}
            Y = {(0,0): [(0,0,1),(0,2,1)], (0,1): [(0,1,1),(0,3,1)],
                 (1,0): [(1,0,1),(1,2,1)], (1,1): [(1,1,1),(1,3,1)]}
            out = {0: [(1,2,1)], 1: [(1,3,1)]}
        elif s == 6:  # M7 = (A12-A22)(B21+B22)
            X = {0: [(0,2,1),(1,2,-1)], 1: [(0,3,1),(1,3,-1)]}
            Y = {(0,0): [(2,0,1),(2,2,1)], (0,1): [(2,1,1),(2,3,1)],
                 (1,0): [(3,0,1),(3,2,1)], (1,1): [(3,1,1),(3,3,1)]}
            out = {0: [(0,0,1),(1,2,1)], 1: [(0,1,1),(1,3,1)]}

        for j in range(2):
            for k in range(2):
                for (i, a, coeff) in X[j]:
                    U[i * 4 + a, col] += coeff
                for (b, c, coeff) in Y[(j,k)]:
                    V[b * 4 + c, col] += coeff
                for (i, k_out, coeff) in out[k]:
                    W[k_out * 2 + i, col] += coeff
                col += 1

    return U, V, W

def _build_schoolbook_1x4x4():
    """Schoolbook for 1x4x4: rank 16. U (4,16), V (16,16), W (4,16)."""
    rank = 16
    U = np.zeros((4, rank))
    V = np.zeros((16, rank))
    W = np.zeros((4, rank))
    for j in range(4):
        for k in range(4):
            U[j, j*4+k] = 1
            V[j*4+k, j*4+k] = 1
            W[k, j*4+k] = 1
    return U, V, W

def _combine_3x4x4():
    """
    Combine 2x4x4 (Strassen, rank 28) and 1x4x4 (schoolbook, rank 16) -> rank 44.
    Returns U (12,44), V (16,44), W (12,44) with W index k*3+i.
    """
    U2, V2, W2 = _build_strassen_2x4x4()
    U1, V1, W1 = _build_schoolbook_1x4x4()

    total_rank = 44
    U = np.zeros((12, total_rank))
    V = np.zeros((16, total_rank))
    W = np.zeros((12, total_rank))

    # 2x4x4 part (cols 0-27)
    U[0:8, 0:28] = U2
    V[0:16, 0:28] = V2
    for i in range(2):
        for k in range(4):
            W[k*3+i, 0:28] = W2[k*2+i, :]

    # 1x4x4 part (cols 28-43)
    U[8:12, 28:44] = U1
    V[0:16, 28:44] = V1
    for k in range(4):
        W[k*3+2, 28:44] = W1[k, :]

    return U, V, W

def _reconstruct(U, V, W):
    """Reconstruct tensor from factors."""
    return np.einsum('ic,jc,kc->ijk', U, V, W)

def _error(U, V, W, T):
    """Frobenius error between reconstruction and target tensor."""
    R = _reconstruct(U, V, W)
    return np.linalg.norm(R - T)

def _project_to_rank(U, V, W, T, target_rank, n, m, p):
    """
    Project current factors (possibly rank > target_rank) down to target_rank
    by finding the best rank-`target_rank` approximation of the tensor and
    re-deriving factors via SVD-based tensor decomposition.
    
    Uses the flattening approach: view the tensor as a matrix and use
    iterative SVD to get a low-rank tensor approximation, then extract
    factors via a greedy matching procedure.
    """
    # Flatten tensor to matrix of shape (n*m, m*p*p*n)
    # Actually, better approach: use the CP decomposition via iterative refinement
    # Start with the current factors, truncate to target_rank via SVD of the
    # "coefficient" structure.
    
    # Approach: form the tensor T_approx = sum_c u_c ⊗ v_c ⊗ w_c
    # To reduce rank, we find linear combinations of the current rank-1 terms
    # that approximate the tensor with fewer terms.
    
    # Use a simple approach: randomly select subsets and optimize, or
    # use the fact that we can do SVD on each mode unfolding.
    
    # Mode-1 unfolding: T_1 has shape (n*m, m*p * p*n)
    # T_1[i, (j,k)*(p*n) + l] = T[i,j,k,l]
    # For a CP rank-r tensor, T_1 = U * (V ⊙ W)^T where ⊙ is Khatri-Rao
    
    # Simpler: use iterative hard thresholding on the tensor
    # Or: just truncate the factors by SVD of the "cross" matrices
    
    # Most practical: do a CP-like SVD on the flattened tensor
    T_flat = T.reshape(n*m, m*p*p*n)
    # SVD
    S, Vt = np.linalg.svd(T_flat, full_matrices=False)
    # Take top target_rank singular values
    k = min(target_rank, len(S))
    U_svd = T_flat[:, :k] / S[:k]
    # Now we have a rank-k approximation in mode-1 sense
    # T_approx_1 = U_svd @ Vt[:k, :]
    # Reconstruct tensor
    T_approx = (U_svd * S[:k] * Vt[:k, :]).reshape(n*m, m*p, p*n)
    
    # Now extract CP factors from T_approx via a greedy procedure
    # For each of the k "components", we need u, v, w vectors
    # This is non-trivial; use an alternating approach
    
    # Alternative: use the current factors but find a linear map from R-dim to target_rank-dim
    # that best preserves the tensor.
    
    # Let's use a different strategy: optimize directly with fewer factors
    # initialized from the SVD.
    
    # For the CP extraction, use a simple heuristic:
    # The k-th component in mode-1 SVD corresponds to direction u_k in U_svd.
    # We need to find v_k, w_k such that sum_k u_k ⊗ v_k ⊗ w_k ≈ T_approx
    
    # Use ALS (alternating least squares) to fit CP factors to T_approx
    U_new = np.zeros((n*m, target_rank))
    V_new = np.zeros((m*p, target_rank))
    W_new = np.zeros((p*n, target_rank))
    
    # Initialize from SVD directions
    U_new[:, :k] = U_svd
    # For V and W, use random init scaled
    V_new = np.random.randn(m*p, target_rank) * 0.1
    W_new = np.random.randn(p*n, target_rank) * 0.1
    
    # ALS iterations
    for _ in range(50):
        # Update U: for each c, U[:,c] = (V[:,c] ⊙ W[:,c]) projected onto T's mode-1
        # T_1 = U @ (V⊙W)^T
        # U[:,c] = sum_{j,l} T[:,j,l] * V[j,c]*W[l,c] / sum_{j,l} (V[j,c]*W[l,c])^2
        for c in range(target_rank):
            vw = V_new[:, c] * W_new[:, c]  # shape (m*p*p*n,)
            # T_1[:, c's direction] = sum over (j,l) of T[i,j,l] * vw[j,l]
            T1 = T.reshape(n*m, -1)
            numerator = T1 @ vw  # shape (n*m,)
            denom = np.dot(vw, vw)
            if denom > 1e-12:
                U_new[:, c] = numerator / denom
        
        # Update V: for each c, V[:,c] = (U[:,c] ⊙ W[:,c]) projected onto T's mode-2
        for c in range(target_rank):
            uw = U_new[:, c].reshape(n*m, 1) * W_new[:, c].reshape(1, p*n)
            # T_2 has shape (m*p, n*m * p*n)
            T2 = np.transpose(T, (1, 0, 2)).reshape(m*p, -1)
            uw_flat = uw.reshape(-1)
            numerator = T2 @ uw_flat
            denom = np.dot(uw_flat, uw_flat)
            if denom > 1e-12:
                V_new[:, c] = numerator / denom
        
        # Update W: for each c, W[:,c] = (U[:,c] ⊙ V[:,c]) projected onto T's mode-3
        for c in range(target_rank):
            uv = U_new[:, c].reshape(n*m, 1) * V_new[:, c].reshape(1, m*p)
            # T_3 has shape (p*n, n*m * m*p)
            T3 = np.transpose(T, (2, 0, 1)).reshape(p*n, -1)
            uv_flat = uv.reshape(-1)
            numerator = T3 @ uv_flat
            denom = np.dot(uv_flat, uv_flat)
            if denom > 1e-12:
                W_new[:, c] = numerator / denom
    
    return U_new, V_new, W_new

def _optimize_factors(U, V, W, T, n, m, p, max_iters=200, lr=0.01, deadline=None):
    """
    Optimize factors to minimize reconstruction error using gradient descent.
    U, V, W are the initial factors (already at target rank).
    """
    best_err = _error(U, V, W, T)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    rank = U.shape[1]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Compute gradients
        # R = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # dL/dU[:,c] = (V[:,c] ⊙ W[:,c]) * (R - T) contracted...
        # More precisely:
        # (R - T)[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c] - T[i,j,k]
        # dL/dU[i,c] = sum_{j,k} (R-T)[i,j,k] * V[j,c] * W[k,c]
        #             = sum_{j,k} (R-T)[i,j,k] * (V⊙W)_{j,k,c}
        
        R = _reconstruct(U, V, W)
        E = R - T  # error tensor
        
        # Gradient for U: for each c, grad_U[:,c] = sum over j,k of E[i,j,k]*V[j,c]*W[k,c]
        # = E reshaped to (n*m, m*p*p*n) @ (V[:,c]*W[:,c])
        E_flat = E.reshape(n*m, m*p*p*n)
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        E_T1 = E.reshape(n*m, m*p*p*n)
        E_T2 = np.transpose(E, (1,0,2)).reshape(m*p, n*m*p*n)
        E_T3 = np.transpose(E, (2,0,1)).reshape(p*n, n*m*m*p)
        
        for c in range(rank):
            vw = V[:, c] * W[:, c]  # (m*p*p*n,)
            grad_U[:, c] = E_T1 @ vw
            
            uw = U[:, c].reshape(n*m, 1) * W[:, c].reshape(1, p*n)
            grad_V[:, c] = E_T2 @ uw.reshape(-1)
            
            uv = U[:, c].reshape(n*m, 1) * V[:, c].reshape(1, m*p)
            grad_W[:, c] = E_T3 @ uv.reshape(-1)
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        err = _error(U, V, W, T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-8:
            break
    
    return best_U, best_V, best_W, best_err

def _try_target_rank(target_rank, U_init, V_init, W_init, T, n, m, p, deadline):
    """Try to achieve a valid decomposition at target_rank."""
    if time.time() > deadline - 10:
        return None
    
    # Project to target rank
    U_t, V_t, W_t = _project_to_rank(U_init, V_init, W_init, T, target_rank, n, m, p)
    
    # Check if already valid
    if _validate(U_t, V_t, W_t, n, m, p):
        return U_t, V_t, W_t
    
    # Optimize
    U_opt, V_opt, W_opt, err = _optimize_factors(
        U_t, V_t, W_t, T, n, m, p, 
        max_iters=100, lr=0.05, deadline=deadline
    )
    
    if _validate(U_opt, V_opt, W_opt, n, m, p):
        return U_opt, V_opt, W_opt
    
    return None

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank = n*m*p."""
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
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    T = _multiplication_tensor(n, m, p)
    
    # Start from the rank-44 Strassen-based decomposition
    U44, V44, W44 = _combine_3x4x4()
    
    # Verify the rank-44 decomposition is correct
    if not _validate(U44, V44, W44, n, m, p):
        # Fallback
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    best_result = (U44.tolist(), V44.tolist(), W44.tolist())
    best_rank = 44
    
    # Try to reduce rank: try 43, 42, 41, ...
    # Known lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # But realistically, try a few values
    for target_rank in range(43, 35, -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_target_rank(target_rank, U44, V44, W44, T, n, m, p, deadline)
        if result is not None:
            U_t, V_t, W_t = result
            best_result = (U_t.tolist(), V_t.tolist(), W_t.tolist())
            best_rank = target_rank
            # Don't break; try to go lower
            # But update the starting point
            U44, V44, W44 = U_t, V_t, W_t
    
    # Final validation
    U_f, V_f, W_f = best_result
    if not _validate(U_f, V_f, W_f, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
