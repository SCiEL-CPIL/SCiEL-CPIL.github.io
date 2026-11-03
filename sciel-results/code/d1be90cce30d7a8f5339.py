<<<<<<< SEARCH
def _optimize_decomposition(U, V, W, n, m, p, max_iter=2000, tol=1e-10, lr=0.01, time_limit=None, start_time=None):
    """Optimize the decomposition using gradient descent with adaptive learning rate."""
    if start_time is None:
        start_time = time.time()
    
    current_error = _compute_error(U, V, W, n, m, p)
    best_error = current_error
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Adaptive learning rate
    effective_lr = lr
    prev_error = current_error
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        dU, dV, dW = _compute_gradients(U, V, W, n, m, p)
        
        # Normalize gradients
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm < 1e-12:
            break
        
        # Update
        U -= effective_lr * dU
        V -= effective_lr * dV
        W -= effective_lr * dW
        
        new_error = _compute_error(U, V, W, n, m, p)
        
        # Adaptive learning rate
        if new_error > prev_error * 1.5:
            effective_lr *= 0.5
            # Revert
            U += effective_lr * 2.0 * dU  # Approximate revert
            V += effective_lr * 2.0 * dV
            W += effective_lr * 2.0 * dW
            new_error = _compute_error(U, V, W, n, m, p)
        else:
            # Slowly increase if making progress
            if new_error < prev_error * 0.95:
                effective_lr = min(effective_lr * 1.05, 0.1)
        
        prev_error = new_error
        
        if new_error < best_error:
            best_error = new_error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_error < tol:
            break
    
    return best_U, best_V, best_W, best_error

def _random_search(n, m, p, target_rank, num_restarts=20, time_limit=30, start_time=None):
    """Random search with multiple restarts and gradient optimization."""
    if start_time is None:
        start_time = time.time()
    
    best_result = None
    best_error = float('inf')
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        np.random.seed(12345 + restart * 7919 + target_rank * 100003)
        
        # Initialize with small random values
        U = np.random.randn(n * m, target_rank) * 0.3
        V = np.random.randn(m * p, target_rank) * 0.3
        W = np.random.randn(p * n, target_rank) * 0.3
        
        # Optimize
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining < 1:
            break
        
        U_opt, V_opt, W_opt, error = _optimize_decomposition(
            U, V, W, n, m, p,
            max_iter=500,
            tol=1e-12,
            lr=0.05,
            time_limit=time_remaining,
            start_time=start_time
        )
        
        if error < best_error:
            best_error = error
            best_result = (U_opt, V_opt, W_opt)
        
        if best_error < 1e-10:
            break
    
    return best_result, best_error
=======
def _als_step(U, V, W, n, m, p, rank):
    """Perform one step of Alternating Least Squares."""
    # Update U
    # For each (i,j), U[i*m+j, :] is the variable x.
    # We want to minimize sum_k (1 - sum_c x_c V[j*p+k, c] W[k*n+i, c])^2
    # This is a linear least squares problem: A x = b
    # A is of size (p, rank), b is of size (p,)
    # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
    # b[k] = 1
    
    new_U = np.zeros_like(U)
    for i in range(n):
        for j in range(m):
            A = V[j*p:(j+1)*p, :] * W[:, i] # Broadcasting: V part (p, rank) * W part (p, 1) -> (p, rank)
            # Actually W[:, i] is shape (p,). We need element-wise multiplication along rows.
            # V[j*p:(j+1)*p, :] is (p, rank).
            # W[:, i] is (p,).
            # We want A[k, c] = V[j*p+k, c] * W[k*n+i, c]
            # So A = V_block * W_col[:, np.newaxis]
            A = V[j*p:(j+1)*p, :] * W[:, i][:, np.newaxis]
            b = np.ones(p)
            # Solve A x = b
            try:
                new_U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                new_U[i*m+j, :] = U[i*m+j, :]
    U = new_U
    
    # Update V
    new_V = np.zeros_like(V)
    for j in range(m):
        for k in range(p):
            # Variable is V[j*p+k, :]
            # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
            # A is (n, rank)
            A = U[:, j] * W[k*n:(k+1)*n, :].T # U[:, j] is (n,). W block is (n, rank). Transpose to (rank, n) then multiply?
            # Let's be careful.
            # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
            # U[:, j] is shape (n,).
            # W[k*n:(k+1)*n, :] is shape (n, rank).
            # We want A (n, rank).
            # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
            # This is element-wise multiplication of U_col (n, 1) and W_block (n, rank).
            A = U[:, j][:, np.newaxis] * W[k*n:(k+1)*n, :]
            b = np.ones(n)
            try:
                new_V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                new_V[j*p+k, :] = V[j*p+k, :]
    V = new_V
    
    # Update W
    new_W = np.zeros_like(W)
    for k in range(p):
        for i in range(n):
            # Variable is W[k*n+i, :]
            # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
            # A is (m, rank)
            # U[i*m:(i+1)*m, :] is (m, rank)
            # V[:, k] is (m,).
            # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
            # A = U_block * V_col[:, np.newaxis]
            A = U[i*m:(i+1)*m, :] * V[:, k][:, np.newaxis]
            b = np.ones(m)
            try:
                new_W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                new_W[k*n+i, :] = W[k*n+i, :]
    W = new_W
    
    return U, V, W

def _als_optimize(U, V, W, n, m, p, max_iter=100, time_limit=None, start_time=None, tol=1e-10):
    """Optimize using ALS."""
    if start_time is None:
        start_time = time.time()
    
    rank = U.shape[1]
    current_error = _compute_error(U, V, W, n, m, p)
    best_error = current_error
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        U, V, W = _als_step(U, V, W, n, m, p, rank)
        
        new_error = _compute_error(U, V, W, n, m, p)
        
        if new_error < best_error:
            best_error = new_error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        if best_error < tol:
            break
            
    return best_U, best_V, best_W, best_error

def _random_search(n, m, p, target_rank, num_restarts=20, time_limit=30, start_time=None):
    """Random search with multiple restarts and ALS optimization."""
    if start_time is None:
        start_time = time.time()
    
    best_result = None
    best_error = float('inf')
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        np.random.seed(12345 + restart * 7919 + target_rank * 100003)
        
        # Initialize with random values
        # Scaling matters for ALS convergence sometimes, but usually robust
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Optimize
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining < 1:
            break
        
        # Allocate iterations based on time remaining
        # ALS step is roughly O(n*m*p*rank^2)
        # Estimate steps per second? Just use a fixed number and check time.
        max_iter = 200 
        
        U_opt, V_opt, W_opt, error = _als_optimize(
            U, V, W, n, m, p,
            max_iter=max_iter,
            time_limit=time_remaining,
            start_time=start_time
        )
        
        if error < best_error:
            best_error = error
            best_result = (U_opt, V_opt, W_opt)
        
        if best_error < 1e-10:
            break
    
    return best_result, best_error
>>>>>>> REPLACE

