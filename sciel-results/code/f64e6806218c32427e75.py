<<<<<<< SEARCH
def _try_rank_gradient(n, m, p, R, deadline, seed=42):
    """
    Gradient-based search followed by snapping to half-integers.
    """
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n*m, R) * 0.5
    V = np.random.randn(m*p, R) * 0.5
    W = np.random.randn(p*n, R) * 0.5
    
    lr = 0.01
    best_res = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for step in range(5000):
        if time.time() > deadline:
            break
            
        res = _residual(U, V, W, n, m, p)
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if res < 1e-8:
            # Snap to half-integers
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            if _verify_exact(U_s, V_s, W_s, n, m, p):
                return U_s, V_s, W_s
            break
        
        # Compute gradients using finite differences (expensive but simple)
        # For efficiency, use a few random finite difference steps
        for _ in range(3):
            mat_choice = np.random.randint(0, 3)
            if mat_choice == 0:
                mat = U; rows = n*m
            elif mat_choice == 1:
                mat = V; rows = m*p
            else:
                mat = W; rows = p*n
                
            row = np.random.randint(0, rows)
            col = np.random.randint(0, R)
            
            eps = 0.01
            old_val = mat[row, col]
            mat[row, col] = old_val + eps
            res_plus = _residual(U, V, W, n, m, p)
            mat[row, col] = old_val - eps
            res_minus = _residual(U, V, W, n, m, p)
            mat[row, col] = old_val
            
            grad = (res_plus - res_minus) / (2 * eps)
            mat[row, col] = old_val - lr * grad
            
            # Keep values bounded
            mat[row, col] = np.clip(mat[row, col], -2.0, 2.0)
            
        if time.time() > deadline:
            break
    
    # Final snap
    U_s = np.round(best_U * 2) / 2
    V_s = np.round(best_V * 2) / 2
    W_s = np.round(best_W * 2) / 2
    if _verify_exact(U_s, V_s, W_s, n, m, p):
        return U_s, V_s, W_s
        
    return None
=======
def _try_rank_gradient(n, m, p, R, deadline, seed=42):
    """
    Gradient-based search followed by snapping to half-integers.
    Uses analytical gradient computation for speed.
    """
    np.random.seed(seed)
    
    U = np.random.randn(n*m, R) * 0.5
    V = np.random.randn(m*p, R) * 0.5
    W = np.random.randn(p*n, R) * 0.5
    
    lr = 0.005
    best_res = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def _compute_gradients(U, V, W, n, m, p, R):
        """Compute analytical gradients of residual w.r.t. U, V, W."""
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        W_perm = np.transpose(W3, (1, 0, 2))  # W_perm[i, k, r] = W[k, i, r]
        
        T_calc = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W_perm, optimize=True)
        residual = T_calc - 1.0
        
        # Gradient w.r.t. U[i,j,r] = sum_{k} residual[i,j,k] * V[j,k,r] * W[k,i,r]
        # = einsum('ijk,jkr,ikr->ijr', residual, V3, W_perm)
        grad_U3 = np.einsum('ijk,jkr,ikr->ijr', residual, V3, W_perm, optimize=True)
        
        # Gradient w.r.t. V[j,k,r] = sum_{i} residual[i,j,k] * U[i,j,r] * W[k,i,r]
        # = einsum('ijk,ijr,ikr->jkr', residual, U3, W_perm)
        grad_V3 = np.einsum('ijk,ijr,ikr->jkr', residual, U3, W_perm, optimize=True)
        
        # Gradient w.r.t. W[k,i,r] = sum_{j} residual[i,j,k] * U[i,j,r] * V[j,k,r]
        # = einsum('ijk,ijr,jkr->ikr', residual, U3, V3)
        grad_W_perm = np.einsum('ijk,ijr,jkr->ikr', residual, U3, V3, optimize=True)
        grad_W3 = np.transpose(grad_W_perm, (0, 2, 1))  # Back to (p, n, R)
        
        return grad_U3.reshape(n*m, R), grad_V3.reshape(m*p, R), grad_W3.reshape(p*n, R)
    
    for step in range(3000):
        if time.time() > deadline:
            break
            
        res = _residual(U, V, W, n, m, p)
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if res < 1e-8:
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            if _verify_exact(U_s, V_s, W_s, n, m, p):
                return U_s, V_s, W_s
            break
        
        # Compute analytical gradients
        grad_U, grad_V, grad_W = _compute_gradients(U, V, W, n, m, p, R)
        
        # Gradient step
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Keep values bounded
        U = np.clip(U, -3.0, 3.0)
        V = np.clip(V, -3.0, 3.0)
        W = np.clip(W, -3.0, 3.0)
        
        # Periodically snap to half-integers and check
        if step % 100 == 0:
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            if _verify_exact(U_s, V_s, W_s, n, m, p):
                return U_s, V_s, W_s
    
    # Final snap
    U_s = np.round(best_U * 2) / 2
    V_s = np.round(best_V * 2) / 2
    W_s = np.round(best_W * 2) / 2
    if _verify_exact(U_s, V_s, W_s, n, m, p):
        return U_s, V_s, W_s
        
    return None
>>>>>>> REPLACE

