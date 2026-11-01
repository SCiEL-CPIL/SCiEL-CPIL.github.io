<<<<<<< SEARCH
def _try_lower_rank(n, m, p, target_rank, time_budget, rng):
    """
    Attempt to find a decomposition of rank target_rank using randomized local search.
    Returns None if not found within time_budget.
    """
    start = time.time()
    deadline = start + time_budget
    
    # Initialize with random small values
    def random_init(rank):
        U = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(n * m)]
        V = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(m * p)]
        W = [[rng.gauss(0, 1) for _ in range(rank)] for _ in range(p * n)]
        return U, V, W
    
    def compute_error(U, V, W, rank):
        """Compute the Frobenius norm of the error tensor."""
        err_sq = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    err_sq += (val - 1.0) ** 2
        # Also check off-diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i2, j2, k2) == (i, j, k):
                                    continue
                                a2 = i2 * m + j2
                                b2 = j2 * p + k2
                                c2 = k2 * n + i2
                                val = 0.0
                                for r in range(rank):
                                    val += U[a2][r] * V[b2][r] * W[c2][r]
                                err_sq += val ** 2
        return math.sqrt(err_sq)
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try multiple random restarts
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        U, V, W = random_init(target_rank)
        
        # Gradient-free local search (random perturbations)
        for step in range(200):
            if time.time() > deadline:
                break
            
            err = compute_error(U, V, W, target_rank)
            if err < 1e-8:
                return {"U": U, "V": V, "W": W}
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
            
            # Random perturbation
            lr = 0.1 / (1 + step * 0.01)
            for _ in range(3):
                mat = rng.choice([U, V, W])
                row = rng.randrange(len(mat))
                col = rng.randrange(target_rank)
                mat[row][col] += rng.gauss(0, lr)
        
        if time.time() > deadline:
            break
    
    # Check best found
    if best_U is not None and best_error < 1e-6:
        # Verify with tighter tolerance
        if _validate(best_U, best_V, best_W, n, m, p, tol=1e-6):
            return {"U": best_U, "V": best_V, "W": best_W}
    
    return None
=======
def _try_lower_rank(n, m, p, target_rank, time_budget, rng):
    """
    Attempt to find a decomposition of rank target_rank using gradient-based optimization.
    Returns None if not found within time_budget.
    """
    start = time.time()
    deadline = start + time_budget
    
    target = _build_target_tensor(n, m, p)
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Strategy 1: Structured initialization based on schoolbook with random perturbations
    # Strategy 2: Pure random initialization
    
    for restart in range(30):
        if time.time() > deadline:
            break
        
        if restart % 2 == 0:
            # Structured init: start from schoolbook-like structure
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Assign schoolbook terms to random rank positions
            assignments = list(range(n * m * p))
            rng.shuffle(assignments)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        r = assignments[idx] % target_rank
                        U[a, r] = 1.0
                        V[b, r] = 1.0
                        W[c, r] = 1.0
                        idx += 1
            
            # Add small random noise to break symmetry
            noise_scale = 0.1
            U += noise_scale * np.random.RandomState(restart).randn(*U.shape)
            V += noise_scale * np.random.RandomState(restart + 1000).randn(*V.shape)
            W += noise_scale * np.random.RandomState(restart + 2000).randn(*W.shape)
        else:
            # Random init
            rs = np.random.RandomState(restart * 7 + 42)
            U = rs.randn(n * m, target_rank) * 0.5
            V = rs.randn(m * p, target_rank) * 0.5
            W = rs.randn(p * n, target_rank) * 0.5
        
        # Gradient descent with adaptive learning rate
        lr = 0.01
        best_local_err = float('inf')
        plateau_count = 0
        
        for step in range(500):
            if time.time() > deadline:
                break
            
            err = _compute_error_np(U, V, W, target)
            
            if err < 1e-10:
                # Found exact solution
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
            
            if err < best_local_err:
                best_local_err = err
                plateau_count = 0
            else:
                plateau_count += 1
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Compute gradients
            dU, dV, dW = _compute_gradients_np(U, V, W, target)
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm > 0:
                effective_lr = lr * (err / (err + 1.0))  # Scale by error
                U -= effective_lr * dU
                V -= effective_lr * dV
                W -= effective_lr * dW
            
            # Reduce learning rate if plateauing
            if plateau_count > 20:
                lr *= 0.5
                plateau_count = 0
                if lr < 1e-6:
                    break
        
        if time.time() > deadline:
            break
    
    # Check best found
    if best_U is not None and best_error < 1e-6:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        if _validate(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
>>>>>>> REPLACE

