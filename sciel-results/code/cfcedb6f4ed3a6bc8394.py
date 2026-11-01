<<<<<<< SEARCH
def _try_reduce(U, V, W, n, m, p, target_rank, deadline, n_trials=6):
    """Attempt to find a valid rank-target_rank decomposition."""
    rng = np.random.RandomState(target_rank * 31 + 7)
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return None
    
    # Strategy A: column sampling with perturbation
    for trial in range(n_trials):
        if time.time() > deadline - 20:
            return None
        
        # Weighted column selection based on term "importance"
        norms = np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0)
        weights = norms / (norms.sum() + 1e-15)
        
        cols = rng.choice(current_rank, size=target_rank, replace=False, p=weights)
        
        U_init = U[:, cols].copy() + rng.randn(n*m, target_rank) * 0.05
        V_init = V[:, cols].copy() + rng.randn(m*p, target_rank) * 0.05
        W_init = W[:, cols].copy() + rng.randn(p*n, target_rank) * 0.05
        
        res = _optimize_from_start(U_init, V_init, W_init, n, m, p, deadline, max_iter=500)
        if res is not None:
            return res
    
    # Strategy B: random initialization
    for trial in range(3):
        if time.time() > deadline - 10:
            return None
        
        U_init = rng.randn(n*m, target_rank) * 0.3
        V_init = rng.randn(m*p, target_rank) * 0.3
        W_init = rng.randn(p*n, target_rank) * 0.3
        
        res = _optimize_from_start(U_init, V_init, W_init, n, m, p, deadline, max_iter=400)
        if res is not None:
            return res
    
    # Strategy C: no-perturbation column subset
    for trial in range(2):
        if time.time() > deadline - 5:
            return None
        
        cols = rng.choice(current_rank, size=target_rank, replace=False)
        U_init = U[:, cols].copy()
        V_init = V[:, cols].copy()
        W_init = W[:, cols].copy()
        
        res = _optimize_from_start(U_init, V_init, W_init, n, m, p, deadline, max_iter=600)
        if res is not None:
            return res
    
    return None
=======
def _try_reduce(U, V, W, n, m, p, target_rank, deadline, n_trials=6):
    """Attempt to find a valid rank-target_rank decomposition using multiple strategies."""
    rng = np.random.RandomState(target_rank * 31 + 7)
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return None
    
    # Strategy A: column sampling with perturbation
    for trial in range(n_trials):
        if time.time() > deadline - 20:
            return None
        
        # Weighted column selection based on term "importance"
        norms = np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0)
        weights = norms / (norms.sum() + 1e-15)
        
        cols = rng.choice(current_rank, size=target_rank, replace=False, p=weights)
        
        U_init = U[:, cols].copy() + rng.randn(n*m, target_rank) * 0.05
        V_init = V[:, cols].copy() + rng.randn(m*p, target_rank) * 0.05
        W_init = W[:, cols].copy() + rng.randn(p*n, target_rank) * 0.05
        
        res = _optimize_from_start(U_init, V_init, W_init, n, m, p, deadline, max_iter=500)
        if res is not None:
            return res
    
    # Strategy B: random initialization
    for trial in range(3):
        if time.time() > deadline - 10:
            return None
        
        U_init = rng.randn(n*m, target_rank) * 0.3
        V_init = rng.randn(m*p, target_rank) * 0.3
        W_init = rng.randn(p*n, target_rank) * 0.3
        
        res = _optimize_from_start(U_init, V_init, W_init, n, m, p, deadline, max_iter=400)
        if res is not None:
            return res
    
    # Strategy C: no-perturbation column subset
    for trial in range(2):
        if time.time() > deadline - 5:
            return None
        
        cols = rng.choice(current_rank, size=target_rank, replace=False)
        U_init = U[:, cols].copy()
        V_init = V[:, cols].copy()
        W_init = W[:, cols].copy()
        
        res = _optimize_from_start(U_init, V_init, W_init, n, m, p, deadline, max_iter=600)
        if res is not None:
            return res
    
    # Strategy D: Try to merge pairs of terms that are "close" in structure
    if current_rank - target_rank <= 4:
        for merge_trial in range(3):
            if time.time() > deadline - 8:
                return None
            
            # Pick a random pair of columns to try to merge
            idx = rng.choice(current_rank, size=2, replace=False)
            
            # Remove one column and optimize the remaining
            keep = [i for i in range(current_rank) if i not in idx[1]]
            U_init = U[:, keep].copy() + rng.randn(n*m, target_rank) * 0.1
            V_init = V[:, keep].copy() + rng.randn(m*p, target_rank) * 0.1
            W_init = W[:, keep].copy() + rng.randn(p*n, target_rank) * 0.1
            
            res = _optimize_from_start(U_init, V_init, W_init, n, m, p, deadline, max_iter=800)
            if res is not None:
                return res
    
    return None
>>>>>>> REPLACE

