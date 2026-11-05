<<<<<<< SEARCH
def _numerical_optimization(n, m, p, initial_rank, time_limit=100.0):
    """
    Try to find a lower rank decomposition using numerical optimization.
    Start from schoolbook and try to reduce rank by solving for a lower-rank approximation.
    """
    start_time = time.time()
    
    # Try ranks from 40 down to 20 (reasonable range for 3x3x8)
    for target_rank in range(40, 19, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random or structured values
        np.random.seed(42)
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Use gradient descent to minimize ||T_recon - T_true||^2
        T_true = _build_tensor(n, m, p)
        
        lr = 0.01
        for step in range(500):
            if time.time() - start_time > time_limit:
                break
            
            # Compute current reconstruction
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            error = T_recon - T_true
            
            # Gradients
            grad_U = np.einsum('abc,br,cr->ar', error, V, W)
            grad_V = np.einsum('abc,ar,cr->br', error, U, W)
            grad_W = np.einsum('abc,ar,br->cr', error, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if np.linalg.norm(error) < 1e-8:
                if _verify(U, V, W, n, m, p, tol=1e-7):
                    return U, V, W, target_rank
        
        # Final check
        if _verify(U, V, W, n, m, p, tol=1e-7):
            return U, V, W, target_rank
    
    return None, None, None, None
=======
def _als_decompose(T, n, m, p, rank, max_iters=200, tol=1e-10, time_budget=60.0, seed=42):
    """
    Alternating Least Squares for tensor decomposition.
    T is the target tensor of shape (n*m, m*p, p*n).
    Returns (U, V, W) if converged, else None.
    """
    start = time.time()
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    prev_err = np.inf
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        if it > 5 and abs(prev_err - np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))) < tol:
            break
        
        # Update U: for each a, solve least squares
        for a in range(n * m):
            # T[a, :, :] = U[a, :] @ (V * W)^T element-wise
            # Reshape: for fixed a, we have a matrix of size (m*p, p*n)
            # U[a, r] * sum over r of V[b, r] * W[c, r] = T[a, b, c]
            # For fixed a: T[a, :, :] = sum_r U[a, r] * (V[:, r] outer W[:, r])
            # This is: T_a = U[a, :] @ (V * W) where (V*W)[b, c, r] = V[b,r]*W[c,r]
            # Reshape V*W to (m*p*p*n, rank)
            VW = np.einsum('br,cr->bcr', V, W).reshape(m * p * p * n, rank)
            Ta = T[a].reshape(-1)
            # Solve: VW^T @ U[a, :]^T = Ta
            A = VW.T  # (rank, m*p*p*n)
            b = Ta
            U[a, :] = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Update V
        for b in range(m * p):
            UW = np.einsum('ar,cr->acr', U, W).reshape(n * m * p * n, rank)
            Tb = T[:, b, :].reshape(-1)
            A = UW.T
            b_vec = Tb
            V[b, :] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
        
        # Update W
        for c in range(p * n):
            UV = np.einsum('ar,br->abr', U, V).reshape(n * m * m * p, rank)
            Tc = T[:, :, c].reshape(-1)
            A = UV.T
            b_vec = Tc
            W[c, :] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
        
        prev_err = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
    
    err = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
    if err < 1e-6:
        return U, V, W
    return None

def _try_rank_with_als(n, m, p, rank, time_budget=60.0, n_restarts=5):
    """Try to find a decomposition of given rank using ALS with multiple restarts."""
    start = time.time()
    T = _build_tensor(n, m, p)
    
    for restart in range(n_restarts):
        if time.time() - start > time_budget:
            break
        result = _als_decompose(T, n, m, p, rank, max_iters=100, tol=1e-10, 
                               time_budget=time_budget - (time.time() - start), 
                               seed=42 + restart)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-7):
                return U, V, W, rank
    return None, None, None, None

def _numerical_optimization(n, m, p, initial_rank, time_limit=100.0):
    """
    Try to find a lower rank decomposition using numerical optimization.
    """
    start_time = time.time()
    
    # Try ranks from 35 down to 20 (reasonable range for 3x3x8)
    # Based on known bounds, rank is likely between 20-30 for this problem
    for target_rank in range(35, 19, -1):
        if time.time() - start_time > time_limit:
            break
        
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
            
        # Use ALS with multiple restarts
        U, V, W, rank = _try_rank_with_als(n, m, p, target_rank, 
                                           time_budget=min(remaining_time, 30.0),
                                           n_restarts=3)
        if rank is not None and rank < initial_rank:
            return U, V, W, rank
    
    return None, None, None, None
>>>>>>> REPLACE

