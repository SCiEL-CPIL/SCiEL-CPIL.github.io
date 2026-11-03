import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_target(n, m, p):
    """Return target tensor as 3D array T[i, j, k] = 1 for all valid indices."""
    return np.ones((n, m, p), dtype=np.float64)

def _tensor_to_matrices(T, n, m, p):
    """
    Reshape target T[i,j,k] into the three matrix views needed for verification:
    U_view[i*m+j, k] = T[i,j,k]  (but we actually want the decomposition to match
    the contraction sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = T[i,j,k])
    We'll just use T directly as the target for the AM algorithm.
    """
    return T

def _am_decompose(T, rank, n, m, p, max_iter=200, tol=1e-12, seed=0):
    """
    Alternating Minimization for 3-way tensor decomposition.
    T is (n, m, p). We seek U (n*m, rank), V (m*p, rank), W (p*n, rank)
    such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k].
    
    This is a non-convex problem; AM cycles through fixing two factors and solving
    for the third via least squares.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize randomly
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Normalize to avoid scale issues
    for c in range(rank):
        scale = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
        if scale > 1e-15:
            U[:, c] /= scale ** (1/3)
            V[:, c] /= scale ** (1/3)
            W[:, c] /= scale ** (1/3)
    
    prev_err = np.inf
    
    for it in range(max_iter):
        # Update U: for each column c, solve least squares
        # Fix V and W. For each c, U[:,c] should satisfy:
        # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = T[i,j,k]
        # This is: U * diag(V_k * W_k) ... it's coupled across k.
        # Better approach: reshape into a matrix problem.
        
        # For fixed V, W: the equation is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
        # Let's define for each (j,k): B_jk[i, c] = V[j*p+k, c] * W[k*n+i, c]
        # Then: sum_c U[i*m+j, c] * B_jk[i, c] = T[i,j,k]
        # This is not a simple LS because U and B share the index c in a product.
        
        # Alternative: use the standard AM for 3-tensor.
        # Reshape T into a matrix: T_mat[i, (j,k)] = T[i,j,k]
        # Then T_mat ≈ sum_c u_c ⊗ (v_c ⊙ w_c) where v_c is length m*p, w_c is length p*n
        # But v_c and w_c are not independent - they're linked by the index structure.
        
        # Let me use a different formulation. Define:
        # A[i, j, k] = T[i,j,k]
        # We want: A[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # For updating U: fix V, W. For each row (i,j) of U, we need:
        # U[i*m+j, :] such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] for all k
        # Let M_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
        # Then: M_ij @ U[i*m+j, :] = T[i,j,:]
        # So U[i*m+j, :] = M_ij^{-1} @ T[i,j,:] (or LS if not square)
        
        # Similarly for V and W.
        
        # Update U
        for i in range(n):
            for j in range(m):
                # Build M: shape (p, rank)
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve M @ x = T[i,j,:]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, T[i, j, :], rcond=None)
                except:
                    continue
                U[i * m + j, :] = x
        
        # Update V
        for j in range(m):
            for k in range(p):
                # Build M: shape (n, rank)
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                # Solve M @ x = T[:,j,k]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, T[:, j, k], rcond=None)
                except:
                    continue
                V[j * p + k, :] = x
        
        # Update W
        for k in range(p):
            for i in range(n):
                # Build M: shape (m, rank)
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                # Solve M @ x = T[i,:,k]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, T[i, :, k], rcond=None)
                except:
                    continue
                W[k * n + i, :] = x
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (s - T[i,j,k])**2
        err = np.sqrt(err)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def _try_round_and_verify(U, V, W, n, m, p, tol=1e-9):
    """
    Try to round the floating-point decomposition to integers/half-integers
    and verify exactness.
    """
    # Try rounding to nearest integer
    U_int = np.round(U).astype(np.int64)
    V_int = np.round(V).astype(np.int64)
    W_int = np.round(W).astype(np.int64)
    
    # Verify with integer arithmetic (exact)
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(len(U_int[0])):
                    s += int(U_int[i*m+j, c]) * int(V_int[j*p+k, c]) * int(W_int[k*n+i, c])
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    # Try half-integer rounding: multiply by 2, round, then divide by 2
    U_half = np.round(U * 2) / 2.0
    V_half = np.round(V * 2) / 2.0
    W_half = np.round(W * 2) / 2.0
    
    # Convert to integers by multiplying by 2
    U_int2 = (U_half * 2).astype(np.int64)
    V_int2 = (V_half * 2).astype(np.int64)
    W_int2 = (W_half * 2).astype(np.int64)
    
    # Verify: sum_c (U_int2/2) * (V_int2/2) * (W_int2/2) = 1
    # => sum_c U_int2 * V_int2 * W_int2 = 8
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(len(U_int2[0])):
                    s += int(U_int2[i*m+j, c]) * int(V_int2[j*p+k, c]) * int(W_int2[k*n+i, c])
                if s != 8:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
    
    return None

def _schoolbook(n, m, p):
    """Return schoolbook rank-n*m*p decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    T = _build_target(n, m, p)
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try decreasing ranks from a reasonable upper bound
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Upper bound: schoolbook = 72
    # Try ranks: 72, 70, 68, ..., down to 24
    # But that's too many. Let's try a smarter set.
    
    # Known: Strassen-style block decomposition gives rank 66 (from knowledge)
    # Let's try ranks from 72 down to 24, but skip some
    # Actually, let's try: 72, 70, 68, 66, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24
    
    ranks_to_try = [72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:  # Leave 20s for finalization
            break
        
        if rank >= best_rank:
            continue  # No point trying ranks >= current best
        
        # Try multiple random restarts for this rank
        max_restarts = 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            seed = restart * 1000 + rank
            U, V, W, err = _am_decompose(T, rank, n, m, p, max_iter=100, tol=1e-10, seed=seed)
            
            # Check if AM converged to near-zero error
            if err < 1e-8:
                # Try to round and verify
                result = _try_round_and_verify(U, V, W, n, m, p)
                if result is not None:
                    U_list, V_list, W_list = result
                    # Final verification
                    if verify(U_list, V_list, W_list, n, m, p):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
            else:
                # Even if AM didn't converge perfectly, try rounding
                # (sometimes AM gets close enough)
                if err < 0.1:
                    result = _try_round_and_verify(U, V, W, n, m, p)
                    if result is not None:
                        U_list, V_list, W_list = result
                        if verify(U_list, V_list, W_list, n, m, p):
                            best = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = rank
                            found = True
                            break
        
        if found:
            # Found a valid decomposition at this rank; try lower ranks
            continue
    
    return best

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    T = _build_target(n, m, p)

    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    ranks_to_try = [72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24]

    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            continue

        max_restarts = 5
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            seed = restart * 1000 + rank
            U, V, W, err = _am_decompose(T, rank, n, m, p, max_iter=100, tol=1e-10, seed=seed)
            if err < 1e-8:
                result = _try_round_and_verify(U, V, W, n, m, p)
                if result is not None:
                    U_list, V_list, W_list = result
                    if verify(U_list, V_list, W_list, n, m, p):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
            elif err < 0.1:
                result = _try_round_and_verify(U, V, W, n, m, p)
                if result is not None:
                    U_list, V_list, W_list = result
                    if verify(U_list, V_list, W_list, n, m, p):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
        if found:
            continue

    return best
# EVOLVE-BLOCK-END
