import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half(arr):
    return np.round(arr * 2) / 2.0

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _compute_loss_and_grad(U, V, W, n, m, p):
    """Compute loss and gradients efficiently."""
    r = U.shape[1]
    loss = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                diff = 2.0 * (val - 1.0)
                loss += 0.5 * diff * diff
                
                # Gradients
                vw = V[v_idx, :] * W[w_idx, :]
                grad_U[u_idx, :] += diff * vw
                grad_V[v_idx, :] += diff * (U[u_idx, :] * W[w_idx, :])
                grad_W[w_idx, :] += diff * (U[u_idx, :] * V[v_idx, :])
    
    return loss, grad_U, grad_V, grad_W

def _polish(U, V, W, n, m, p, max_iter=1000, tol=1e-12):
    """Final high-precision polish using Gauss-Newton style updates."""
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for _ in range(max_iter):
        loss, grad_U, grad_V, grad_W = _compute_loss_and_grad(U, V, W, n, m, p)
        if loss < tol:
            break
        
        # Simple gradient descent with adaptive step size
        grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if grad_norm < 1e-15:
            break
        
        step_size = min(0.1, 1.0 / (grad_norm + 1e-10))
        
        U_new = U - step_size * grad_U
        V_new = V - step_size * grad_V
        W_new = W - step_size * grad_W
        
        # Check if loss decreased
        loss_new, _, _, _ = _compute_loss_and_grad(U_new, V_new, W_new, n, m, p)
        
        if loss_new < loss:
            U, V, W = U_new, V_new, W_new
        else:
            # Halve step size
            step_size /= 2
            if step_size < 1e-15:
                break
    
    return U, V, W

def _optimize_rank(r, n, m, p, time_limit=10):
    start_time = time.time()
    
    def unpack(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)
        return U, V, W
    
    def objective(x):
        U, V, W = unpack(x)
        loss, _, _, _ = _compute_loss_and_grad(U, V, W, n, m, p)
        return loss
    
    def gradient(x):
        U, V, W = unpack(x)
        _, gU, gV, gW = _compute_loss_and_grad(U, V, W, n, m, p)
        return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
    
    best_loss = float('inf')
    best_x = None
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Small random
    initializations.append(np.random.randn(n*m*r + m*p*r + p*n*r) * 0.1)
    
    # 2. Structured: initialize based on schoolbook pattern with noise
    U_init = np.zeros((n*m, r))
    V_init = np.zeros((m*p, r))
    W_init = np.zeros((p*n, r))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = (i*m+j)*p + k
                if col < r:
                    U_init[i*m+j, col] = 1
                    V_init[j*p+k, col] = 1
                    W_init[k*n+i, col] = 1
    # Add some random columns
    for c in range(r):
        if c >= n*m*p:
            U_init[:, c] = np.random.randn(n*m) * 0.2
            V_init[:, c] = np.random.randn(m*p) * 0.2
            W_init[:, c] = np.random.randn(p*n) * 0.2
    initializations.append(np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()]))
    
    # 3. Another structured variant
    U_init = np.zeros((n*m, r))
    V_init = np.zeros((m*p, r))
    W_init = np.zeros((p*n, r))
    for c in range(r):
        U_init[:, c] = np.random.randn(n*m)
        V_init[:, c] = np.random.randn(m*p)
        W_init[:, c] = np.random.randn(p*n)
    # Normalize to help with optimization
    U_init /= np.linalg.norm(U_init, axis=0, keepdims=True) + 1e-10
    V_init /= np.linalg.norm(V_init, axis=0, keepdims=True) + 1e-10
    W_init /= np.linalg.norm(W_init, axis=0, keepdims=True) + 1e-10
    initializations.append(np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()]))
    
    # Try each initialization
    for init_idx, x0 in enumerate(initializations):
        if time.time() - start_time > time_limit:
            break
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x.copy()
                
                U, V, W = unpack(best_x)
                
                # Polish
                U, V, W = _polish(U, V, W, n, m, p, max_iter=200, tol=1e-14)
                
                # Snap to half-integers
                U_snap = _snap_to_half(U)
                V_snap = _snap_to_half(V)
                W_snap = _snap_to_half(W)
                
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-8):
                    return U_snap, V_snap, W_snap, True
                    
                # If snapping fails, try without snapping but check if close enough
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
                    # Try to find nearby half-integer solution
                    U_half = _snap_to_half(U)
                    V_half = _snap_to_half(V)
                    W_half = _snap_to_half(W)
                    if _verify_decomposition(U_half, V_half, W_half, n, m, p, tol=1e-6):
                        return U_half, V_half, W_half, True
                        
        except Exception:
            pass
    
    # Final check on best solution found
    if best_x is not None and best_loss < 1e-8:
        U, V, W = unpack(best_x)
        U, V, W = _polish(U, V, W, n, m, p, max_iter=500, tol=1e-15)
        
        U_snap = _snap_to_half(U)
        V_snap = _snap_to_half(V)
        W_snap = _snap_to_half(W)
        
        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-8):
            return U_snap, V_snap, W_snap, True
            
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from just below schoolbook down to lower bound
    # This ensures we always have a valid solution (schoolbook) and try to improve it
    ranks_to_try = list(range(n*m*p - 1, lower_bound - 1, -1))
    
    for r in ranks_to_try:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 3:
            break
        
        # Allocate time based on rank: smaller ranks get less time as we run out of time
        # Prioritize higher ranks (closer to schoolbook) as they are more likely to exist
        time_alloc = min(10, remaining * 0.5)
        
        U, V, W, success = _optimize_rank(r, n, m, p, time_limit=time_alloc)
        
        if success:
            best_U, best_V, best_W = U, V, W
            # Continue searching for even lower ranks
            
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    return {"U": U_list, "V": V_list, "W": W_list}
