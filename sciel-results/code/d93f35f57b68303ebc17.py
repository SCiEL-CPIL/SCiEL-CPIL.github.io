import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k.
    """
    rank = U.shape[1]
    T_decomposed = np.zeros((n, m, p))
    for c in range(rank):
        u_mat = U[:, c].reshape(n, m)
        v_mat = V[:, c].reshape(m, p)
        w_mat = W[:, c].reshape(p, n)
        for i in range(n):
            temp = u_mat[i, :][:, np.newaxis] * v_mat
            T_decomposed[i, :, :] = temp * w_mat[:, i][np.newaxis, :]
    return np.allclose(T_decomposed, 1.0, atol=tol)

def _try_eliminate_term(U, V, W, c_remove, n, m, p):
    """
    Try to eliminate term c_remove by expressing it as a linear combination of other terms.
    Returns (success, new_U, new_V, new_W) if successful.
    """
    rank = U.shape[1]
    if rank <= 1:
        return False, U, V, W
    
    remaining = [c for c in range(rank) if c != c_remove]
    
    # Build linear system: find coefficients alpha_c such that
    # sum_{c in remaining} alpha_c * (U[:,c], V[:,c], W[:,c]) = (U[:,c_remove], V[:,c_remove], W[:,c_remove])
    # This is an overdetermined system. We'll use least squares.
    
    # Stack the vectors
    u_target = U[:, c_remove]
    v_target = V[:, c_remove]
    w_target = W[:, c_remove]
    
    # Build matrix where each column is the concatenation of (U[:,c], V[:,c], W[:,c]) for c in remaining
    A = np.zeros((n*m + m*p + p*n, len(remaining)))
    for idx, c in enumerate(remaining):
        A[:n*m, idx] = U[:, c]
        A[n*m:n*m+m*p, idx] = V[:, c]
        A[n*m+m*p:, idx] = W[:, c]
    
    b = np.concatenate([u_target, v_target, w_target])
    
    # Solve least squares
    try:
        alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return False, U, V, W
    
    # Check if the solution is good
    residual = np.linalg.norm(A @ alpha - b)
    if residual > 1e-8:
        return False, U, V, W
    
    # Check if coefficients are "nice" (integer or half-integer)
    # For exact decomposition, we need exact coefficients
    # Try rounding to nearest 0.5 and check
    alpha_rounded = np.round(alpha * 2) / 2
    residual_rounded = np.linalg.norm(A @ alpha_rounded - b)
    
    if residual_rounded > 1e-10:
        return False, U, V, W
    
    # Create new matrices without the removed term and with adjusted coefficients
    new_rank = len(remaining)
    new_U = np.zeros((n*m, new_rank))
    new_V = np.zeros((m*p, new_rank))
    new_W = np.zeros((p*n, new_rank))
    
    for idx, c in enumerate(remaining):
        new_U[:, idx] = U[:, c] * alpha_rounded[idx]
        new_V[:, idx] = V[:, c] * alpha_rounded[idx]
        new_W[:, idx] = W[:, c] * alpha_rounded[idx]
    
    return True, new_U, new_V, new_W

def _greedy_rank_reduction(U, V, W, n, m, p, time_limit=60):
    """
    Greedily try to eliminate terms one by one.
    Returns (best_U, best_V, best_W, best_rank).
    """
    start_time = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = U.shape[1]
    
    # Try eliminating each term in order
    for c_remove in range(current_rank):
        if time.time() - start_time > time_limit:
            break
        
        success, new_U, new_V, new_W = _try_eliminate_term(current_U, current_V, current_W, c_remove, n, m, p)
        
        if success:
            # Verify the new decomposition
            if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                current_U, current_V, current_W = new_U, new_V, new_W
                current_rank = new_U.shape[1]
                if current_rank < best_rank:
                    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                    best_rank = current_rank
                # Continue trying to eliminate more terms
                # Re-index: we removed term c_remove, so adjust indices
                # Actually, we need to restart the elimination process since indices changed
                break
    
    # If we made progress, try again from the new state
    if best_rank < U.shape[1]:
        start_time = time.time()
        current_U, current_V, current_W = best_U.copy(), best_V.copy(), best_W.copy()
        current_rank = best_rank
        
        for c_remove in range(current_rank):
            if time.time() - start_time > time_limit:
                break
            
            success, new_U, new_V, new_W = _try_eliminate_term(current_U, current_V, current_W, c_remove, n, m, p)
            
            if success:
                if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                    current_U, current_V, current_W = new_U, new_V, new_W
                    current_rank = new_U.shape[1]
                    if current_rank < best_rank:
                        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
                        best_rank = current_rank
                    break
    
    return best_U, best_V, best_W, best_rank

def _schoolbook_decomposition(n, m, p):
    """Return (U, V, W) for the schoolbook rank n*m*p decomposition."""
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

def construct():
    deadline = time.time() + 115.0  # 115 second deadline (5s margin for return)
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 105)
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Verify the initial decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Initial schoolbook decomposition is invalid")
    
    # Try greedy rank reduction
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        search_time = min(60, time_remaining - 5)
        U_red, V_red, W_red, rank_red = _greedy_rank_reduction(U, V, W, n, m, p, time_limit=search_time)
        
        if rank_red < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = rank_red
    
    # Return the best decomposition found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
