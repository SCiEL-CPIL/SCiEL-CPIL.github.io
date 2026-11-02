# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Return the matrix multiplication tensor T of shape (n, m, p) where T[i,j,k]=1."""
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U[i*m + j, :]
                v_vec = V[j*p + k, :]
                w_vec = W[k*n + i, :]
                val = np.sum(u_vec * v_vec * w_vec)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _random_decomposition(n, m, p, rank, seed=None):
    """Generate a random rank decomposition."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
        
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    return U, V, W

def _compute_error(U, V, W, n, m, p):
    """Compute the squared error of the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err += (val - 1.0)**2
    return err

def _optimize_decomposition(U, V, W, n, m, p, time_budget, start_time):
    """Optimize the decomposition using Nelder-Mead."""
    from scipy.optimize import minimize
    
    def flatten_params():
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
    def unpack_params(x):
        U_flat = x[:n*m*U.shape[1]]
        V_flat = x[n*m*U.shape[1]:n*m*U.shape[1] + m*p*V.shape[1]]
        W_flat = x[n*m*U.shape[1] + m*p*V.shape[1]:]
        
        U_new = U_flat.reshape(n*m, U.shape[1])
        V_new = V_flat.reshape(m*p, V.shape[1])
        W_new = W_flat.reshape(p*n, W.shape[1])
        return U_new, V_new, W_new
        
    def objective(x):
        U_new, V_new, W_new = unpack_params(x)
        return _compute_error(U_new, V_new, W_new, n, m, p)
        
    x0 = flatten_params()
    best_f = objective(x0)
    best_x = x0
    
    try:
        # Use Nelder-Mead
        # Adaptive maxiter based on remaining time? 
        # Or just a fixed reasonable number.
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-10})
        if res.fun < best_f:
            best_x = res.x
            best_f = res.fun
    except Exception:
        pass
        
    U_opt, V_opt, W_opt = unpack_params(best_x)
    return U_opt, V_opt, W_opt, best_f

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Schoolbook Decomposition (Rank 36)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    if not _verify_decomposition(U_school, V_school, W_school, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
        
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Search for lower rank
    # Try ranks from 35 down to 18
    for rank in range(rank_school - 1, 17, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        found = False
        for restart in range(10): # 10 restarts per rank
            if time.time() > deadline:
                break
                
            U, V, W = _random_decomposition(n, m, p, rank, seed=restart * 100 + rank)
            
            # Optimize
            U_opt, V_opt, W_opt, err = _optimize_decomposition(U, V, W, n, m, p, deadline - time.time(), start_time)
            
            if err < 1e-8:
                if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = rank
                    found = True
                    break # Move to next lower rank
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
