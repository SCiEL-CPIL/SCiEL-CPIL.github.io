<<<<<<< SEARCH
def _try_rank_r(T, r, max_iters=50, seed=42, deadline=None):
    """
    Try to find a rank-r decomposition using ALS with random restarts.
    Returns (U, V, W) as numpy arrays if successful, else None.
    """
    np.random.seed(seed)
    N, P, Q = T.shape  # 4, 20, 20
    
    best_error = float('inf')
    best_solution = None
    
    for attempt in range(20):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(N, r) * 0.1
        V = np.random.randn(P, r) * 0.1
        W = np.random.randn(Q, r) * 0.1
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U using least squares
            A = np.zeros((P * Q, r))
            for col in range(r):
                A[:, col] = (V[:, col] * W[:, col][:, None]).flatten()
            
            try:
                U = np.linalg.lstsq(A.T, T.reshape(N, P * Q).T, rcond=None)[0].T
            except:
                continue
            
            # Update V
            B = np.zeros((N * Q, r))
            for col in range(r):
                B[:, col] = (U[:, col] * W[:, col][:, None]).flatten()
            try:
                V = np.linalg.lstsq(B.T, T.reshape(P, N * Q).T, rcond=None)[0].T
            except:
                continue
            
            # Update W
            C = np.zeros((N * P, r))
            for col in range(r):
                C[:, col] = (U[:, col] * V[:, col][:, None]).flatten()
            try:
                W = np.linalg.lstsq(C.T, T.reshape(Q, N * P).T, rcond=None)[0].T
            except:
                continue
            
            # Check convergence
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(recon - T)
            if error < 1e-10:
                break
        
        if deadline is not None and time.time() > deadline:
            break
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        recon = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
        error = np.linalg.norm(recon - T)
        
        if error < best_error:
            best_error = error
            best_solution = (U_rounded, V_rounded, W_rounded)
        
        if error < 1e-6:
            break
    
    if best_error < 1e-6:
        return best_solution
    return None
=======
def _refine_half_integer(U, V, W, T, max_iters=100, tol=1e-10):
    """
    Refine a near-half-integer solution by projecting to half-integers and doing local search.
    Returns (U, V, W) if successful, else None.
    """
    N, P, Q = T.shape
    r = U.shape[1]
    
    # Start from rounded values
    U_curr = np.round(U * 2) / 2
    V_curr = np.round(V * 2) / 2
    W_curr = np.round(W * 2) / 2
    
    for it in range(max_iters):
        recon = np.einsum('ir,jr,kr->ijk', U_curr, V_curr, W_curr)
        error = np.linalg.norm(recon - T)
        if error < tol:
            return (U_curr, V_curr, W_curr)
        
        # Try flipping individual entries to half-integers
        best_improvement = None
        best_new = None
        best_err = error
        
        # Try perturbing U
        for i in range(N):
            for c in range(r):
                old = U_curr[i, c]
                for delta in [-1.0, 1.0]:
                    U_curr[i, c] = old + delta
                    recon = np.einsum('ir,jr,kr->ijk', U_curr, V_curr, W_curr)
                    err = np.linalg.norm(recon - T)
                    if err < best_err - 1e-12:
                        best_err = err
                        best_improvement = ('U', i, c, delta)
                        best_new = (U_curr.copy(), V_curr.copy(), W_curr.copy())
                    U_curr[i, c] = old
        
        # Try perturbing V
        for j in range(P):
            for c in range(r):
                old = V_curr[j, c]
                for delta in [-1.0, 1.0]:
                    V_curr[j, c] = old + delta
                    recon = np.einsum('ir,jr,kr->ijk', U_curr, V_curr, W_curr)
                    err = np.linalg.norm(recon - T)
                    if err < best_err - 1e-12:
                        best_err = err
                        best_improvement = ('V', j, c, delta)
                        best_new = (U_curr.copy(), V_curr.copy(), W_curr.copy())
                    V_curr[j, c] = old
        
        # Try perturbing W
        for k in range(Q):
            for c in range(r):
                old = W_curr[k, c]
                for delta in [-1.0, 1.0]:
                    W_curr[k, c] = old + delta
                    recon = np.einsum('ir,jr,kr->ijk', U_curr, V_curr, W_curr)
                    err = np.linalg.norm(recon - T)
                    if err < best_err - 1e-12:
                        best_err = err
                        best_improvement = ('W', k, c, delta)
                        best_new = (U_curr.copy(), V_curr.copy(), W_curr.copy())
                    W_curr[k, c] = old
        
        if best_improvement is None:
            break
        
        U_curr, V_curr, W_curr = best_new
    
    recon = np.einsum('ir,jr,kr->ijk', U_curr, V_curr, W_curr)
    error = np.linalg.norm(recon - T)
    if error < tol:
        return (U_curr, V_curr, W_curr)
    return None

def _try_rank_r(T, r, max_iters=50, seed=42, deadline=None):
    """
    Try to find a rank-r decomposition using ALS with random restarts.
    Returns (U, V, W) as numpy arrays if successful, else None.
    """
    np.random.seed(seed)
    N, P, Q = T.shape  # 4, 20, 20
    
    best_error = float('inf')
    best_solution = None
    
    for attempt in range(20):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(N, r) * 0.1
        V = np.random.randn(P, r) * 0.1
        W = np.random.randn(Q, r) * 0.1
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U using least squares
            A = np.zeros((P * Q, r))
            for col in range(r):
                A[:, col] = (V[:, col] * W[:, col][:, None]).flatten()
            
            try:
                U = np.linalg.lstsq(A.T, T.reshape(N, P * Q).T, rcond=None)[0].T
            except:
                continue
            
            # Update V
            B = np.zeros((N * Q, r))
            for col in range(r):
                B[:, col] = (U[:, col] * W[:, col][:, None]).flatten()
            try:
                V = np.linalg.lstsq(B.T, T.reshape(P, N * Q).T, rcond=None)[0].T
            except:
                continue
            
            # Update W
            C = np.zeros((N * P, r))
            for col in range(r):
                C[:, col] = (U[:, col] * V[:, col][:, None]).flatten()
            try:
                W = np.linalg.lstsq(C.T, T.reshape(Q, N * P).T, rcond=None)[0].T
            except:
                continue
            
            # Check convergence
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(recon - T)
            if error < 1e-10:
                break
        
        if deadline is not None and time.time() > deadline:
            break
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        recon = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
        error = np.linalg.norm(recon - T)
        
        if error < best_error:
            best_error = error
            best_solution = (U_rounded, V_rounded, W_rounded)
        
        # Try local refinement if close
        if error < 1.0:
            refined = _refine_half_integer(U, V, W, T, max_iters=50)
            if refined is not None:
                U_r, V_r, W_r = refined
                recon_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                error_r = np.linalg.norm(recon_r - T)
                if error_r < 1e-6:
                    return (U_r, V_r, W_r)
        
        if error < 1e-6:
            break
    
    if best_error < 1e-6:
        return best_solution
    return None
>>>>>>> REPLACE

