# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill
    
    # Schoolbook as fallback
    def schoolbook():
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
    
    best = schoolbook()
    best_rank = n * m * p
    
    def check_valid_3d(U, V, W, tol=1e-6):
        """Check if 3D decomposition is valid."""
        prod = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        return np.allclose(prod, 1.0, atol=tol)

    def try_rank_gradient(R, max_iters=2000, seed=None):
        """Try to find a valid decomposition of rank R using gradient descent."""
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        # Initialize with small random values
        U = rng.standard_normal((n, m, R)) * 0.5
        V = rng.standard_normal((m, p, R)) * 0.5
        W = rng.standard_normal((p, n, R)) * 0.5
        
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        best_loss = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute tensor product
            prod = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            error = prod - 1.0
            loss = np.sum(error**2)
            
            if loss < best_loss:
                best_loss = loss
                if loss < 1e-8:
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if check_valid_3d(U_snap, V_snap, W_snap, tol=1e-6):
                        return U_snap.reshape(n*m, R).tolist(), V_snap.reshape(m*p, R).tolist(), W_snap.reshape(p*n, R).tolist()
            
            # Compute gradients
            dU = 2.0 * np.einsum('ijk,jkr,kir->ijr', error, V, W)
            dV = 2.0 * np.einsum('ijk,ijr,kir->jkr', error, U, W)
            dW = 2.0 * np.einsum('ijk,ijr,jkr->kir', error, U, V)
            
            # Update with momentum
            vU = momentum * vU + (1 - momentum) * dU
            vV = momentum * vV + (1 - momentum) * dV
            vW = momentum * vW + (1 - momentum) * dW
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
            
            if it % 100 == 0 and loss < 1e-4:
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if check_valid_3d(U_snap, V_snap, W_snap, tol=1e-5):
                    return U_snap.reshape(n*m, R).tolist(), V_snap.reshape(m*p, R).tolist(), W_snap.reshape(p*n, R).tolist()
        
        return None
    
    # Try ranks from low to high
    for R in range(40, min(60, best_rank)):
        if time.time() > deadline - 10:
            break
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            result = try_rank_gradient(R, max_iters=1000, seed=R*100+attempt)
            if result is not None:
                U, V, W = result
                best = {"U": U, "V": V, "W": W}
                best_rank = R
                break
    
    return best
# EVOLVE-BLOCK-END
