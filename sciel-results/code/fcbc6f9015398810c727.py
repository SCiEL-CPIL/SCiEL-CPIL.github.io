<<<<<<< SEARCH
    def try_lower_rank_random():
        """Attempt to find a lower rank decomposition via random initialization."""
        if time.time() > deadline:
            return None
        
        best_rank = None
        best_decomp = None
        
        # Try ranks from 48 down to 42
        for target_rank in [48, 47, 46, 45, 44]:
            if time.time() > deadline:
                break
            
            # Try a few random initializations
            for attempt in range(10):
                if time.time() > deadline:
                    break
                
                np.random.seed(42 + target_rank * 100 + attempt)
                
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
                
                # Use gradient descent with backtracking line search
                lr = 0.05
                best_loss = np.inf
                for step in range(500):
                    if time.time() > deadline:
                        break
                    
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    error = recon - T
                    loss = np.sum(error**2)
                    
                    if loss < best_loss:
                        best_loss = loss
                        if loss < 1e-10:
                            if verify_decomp(U, V, W):
                                if best_rank is None or target_rank < best_rank:
                                    best_rank = target_rank
                                    best_decomp = (U.copy(), V.copy(), W.copy())
                            break
                    
                    dU = np.einsum('abc,br,cr->ar', error, V, W)
                    dV = np.einsum('abc,ar,cr->br', error, U, W)
                    dW = np.einsum('abc,ar,br->cr', error, U, V)
                    
                    # Backtracking line search
                    step_lr = lr
                    while step_lr > 1e-8:
                        U_new = U - step_lr * dU
                        V_new = V - step_lr * dV
                        W_new = W - step_lr * dW
                        recon_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                        loss_new = np.sum((recon_new - T)**2)
                        if loss_new < loss:
                            U, V, W = U_new, V_new, W_new
                            break
                        step_lr *= 0.5
                    
                if best_rank is not None and target_rank == best_rank:
                    break
        
        return best_decomp
=======
    def try_lower_rank_als():
        """Attempt to find a lower rank decomposition via alternating least squares."""
        if time.time() > deadline:
            return None
        
        best_rank = None
        best_decomp = None
        best_loss = np.inf
        
        # Try ranks from 49 down to 44
        for target_rank in [49, 48, 47, 46, 45, 44]:
            if time.time() > deadline:
                break
            
            # Try multiple random initializations
            for attempt in range(20):
                if time.time() > deadline:
                    break
                
                np.random.seed(42 + target_rank * 1000 + attempt)
                
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
                
                # ALS iterations
                for step in range(200):
                    if time.time() > deadline - 5:
                        break
                    
                    # Update U: for each a, solve U[a,:] from T[a,:,:] = sum_r U[a,r] V[:,r] W[:,r]
                    for a in range(n * m):
                        # T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                        # Reshape to system: for each (b,c), T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                        # This is a linear system in U[a,:]
                        # Build A: A[(b,c), r] = V[b,r] * W[c,r]
                        A = np.einsum('br,cr->bcr', V, W).reshape(-1, target_rank)
                        rhs = T[a].reshape(-1)
                        # Solve least squares
                        try:
                            U[a] = np.linalg.lstsq(A, rhs, rcond=None)[0]
                        except:
                            pass
                    
                    # Update V: for each b, solve V[b,:] from T[:,b,:] = sum_r U[:,r] V[b,r] W[:,r]
                    for b in range(m * p):
                        A = np.einsum('ar,cr->acr', U, W).reshape(-1, target_rank)
                        rhs = T[:, b, :].reshape(-1)
                        try:
                            V[b] = np.linalg.lstsq(A, rhs, rcond=None)[0]
                        except:
                            pass
                    
                    # Update W: for each c, solve W[c,:] from T[:,:,c] = sum_r U[:,r] V[:,r] W[c,r]
                    for c in range(p * n):
                        A = np.einsum('ar,br->abr', U, V).reshape(-1, target_rank)
                        rhs = T[:, :, c].reshape(-1)
                        try:
                            W[c] = np.linalg.lstsq(A, rhs, rcond=None)[0]
                        except:
                            pass
                    
                    # Check convergence
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    loss = np.sum((recon - T)**2)
                    
                    if loss < best_loss:
                        best_loss = loss
                        if loss < 1e-8:
                            if verify_decomp(U, V, W):
                                if best_rank is None or target_rank < best_rank:
                                    best_rank = target_rank
                                    best_decomp = (U.copy(), V.copy(), W.copy())
                            break
                
                if best_rank is not None and target_rank == best_rank:
                    break
        
        return best_decomp
>>>>>>> REPLACE

