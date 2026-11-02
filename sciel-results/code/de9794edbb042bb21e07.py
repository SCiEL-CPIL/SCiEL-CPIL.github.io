<<<<<<< SEARCH
def _try_rank_decomposition(T, n, m, p, target_rank, time_budget=12):
    """
    Try to find a rank-target_rank decomposition of tensor T.
    Uses random initialization + gradient descent on the decomposition parameters.
    """
    dim_u = n * m  # 4
    dim_v = m * p  # 20
    dim_w = p * n  # 20
    
    start_time = time.time()
    
    # Try multiple random initializations
    for trial in range(8):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        U = np.random.randn(dim_u, target_rank) * 0.5
        V = np.random.randn(dim_v, target_rank) * 0.5
        W = np.random.randn(dim_w, target_rank) * 0.5
        
        # Gradient descent with adaptive learning rate
        lr = 0.05
        min_lr = 1e-6
        patience = 50
        no_improve = 0
        best_loss = float('inf')
        
        for step in range(800):
            if time.time() - start_time > time_budget:
                break
            
            # Compute current tensor estimate
            T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = T - T_est
            loss = np.sum(diff**2)
            
            # Track best
            if loss < best_loss:
                best_loss = loss
                no_improve = 0
                U_best = U.copy()
                V_best = V.copy()
                W_best = W.copy()
            else:
                no_improve += 1
            
            if no_improve > patience and lr < min_lr:
                break
            
            # Compute gradients
            grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
            grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
            grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate
            if no_improve > 20:
                lr *= 0.5
                if lr < min_lr:
                    break
            
            # Check convergence
            if loss < 1e-10:
                break
        
        # Final check using best found
        if 'U_best' in locals():
            T_final = np.einsum('ir,jr,kr->ijk', U_best, V_best, W_best)
            if np.max(np.abs(T - T_final)) < 1e-8:
                # Convert to list format
                U_list = U_best.tolist()
                V_list = V_best.tolist()
                W_list = W_best.tolist()
                return U_list, V_list, W_list
    
    return None
=======
def _try_rank_decomposition(T, n, m, p, target_rank, time_budget=8):
    """
    Try to find a rank-target_rank decomposition of tensor T.
    Uses random initialization + gradient descent on the decomposition parameters.
    """
    dim_u = n * m  # 4
    dim_v = m * p  # 20
    dim_w = p * n  # 20
    
    start_time = time.time()
    
    # Try multiple random initializations
    for trial in range(6):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        U = np.random.randn(dim_u, target_rank) * 0.5
        V = np.random.randn(dim_v, target_rank) * 0.5
        W = np.random.randn(dim_w, target_rank) * 0.5
        
        # Gradient descent with adaptive learning rate
        lr = 0.05
        min_lr = 1e-6
        patience = 30
        no_improve = 0
        best_loss = float('inf')
        
        for step in range(500):
            if time.time() - start_time > time_budget:
                break
            
            # Compute current tensor estimate
            T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = T - T_est
            loss = np.sum(diff**2)
            
            # Track best
            if loss < best_loss:
                best_loss = loss
                no_improve = 0
                U_best = U.copy()
                V_best = V.copy()
                W_best = W.copy()
            else:
                no_improve += 1
            
            if no_improve > patience and lr < min_lr:
                break
            
            # Compute gradients
            grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
            grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
            grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate
            if no_improve > 15:
                lr *= 0.5
                if lr < min_lr:
                    break
            
            # Check convergence
            if loss < 1e-10:
                break
        
        # Final check using best found
        if 'U_best' in locals():
            T_final = np.einsum('ir,jr,kr->ijk', U_best, V_best, W_best)
            if np.max(np.abs(T - T_final)) < 1e-8:
                # Convert to list format
                U_list = U_best.tolist()
                V_list = V_best.tolist()
                W_list = W_best.tolist()
                return U_list, V_list, W_list
    
    return None
>>>>>>> REPLACE

