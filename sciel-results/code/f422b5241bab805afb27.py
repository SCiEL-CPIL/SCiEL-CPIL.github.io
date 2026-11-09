import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # 10-second safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Build the target tensor T as a 3D array
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p  # 63
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W form a valid rank-r decomposition."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_rec, T, atol=tol)
    
    def try_gradient_descent(r, max_iters=200, tol=1e-10, seed=42):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline:
            return None
            
        np.random.seed(seed)
        
        # Initialize with random sparse structure
        # Use a structured initialization: start with schoolbook-like patterns
        # but with some merging
        
        # Strategy: initialize U, V, W with values that have some structure
        # For matrix multiplication, we can use the fact that the tensor has
        # a specific pattern
        
        init_U = np.random.randn(nm, r) * 0.5
        init_V = np.random.randn(mp, r) * 0.5
        init_W = np.random.randn(pn, r) * 0.5
        
        def compute_error(U, V, W):
            T_rec = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            return T_rec - T
        
        def objective_and_grad(U, V, W):
            error = compute_error(U, V, W)
            obj = np.sum(error**2)
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r_idx in range(r):
                grad_U[:, r_idx] = 2 * np.einsum('abc,b,c->a', error, V[:, r_idx], W[:, r_idx])
                grad_V[:, r_idx] = 2 * np.einsum('abc,a,c->b', error, U[:, r_idx], W[:, r_idx])
                grad_W[:, r_idx] = 2 * np.einsum('abc,a,b->c', error, U[:, r_idx], V[:, r_idx])
                
            return obj, grad_U, grad_V, grad_W
        
        # Adam optimizer
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        U_curr = init_U.copy()
        V_curr = init_V.copy()
        W_curr = init_W.copy()
        
        best_obj = np.inf
        best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            obj, gU, gV, gW = objective_and_grad(U_curr, V_curr, W_curr)
            
            if obj < best_obj:
                best_obj = obj
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                
            if obj < tol:
                break
                
            # Adam update
            t = it + 1
            m_U = beta1 * m_U + (1 - beta1) * gU
            v_U = beta2 * v_U + (1 - beta2) * gU**2
            m_U_hat = m_U / (1 - beta1**t)
            v_U_hat = v_U / (1 - beta2**t)
            
            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV**2
            m_V_hat = m_V / (1 - beta1**t)
            v_V_hat = v_V / (1 - beta2**t)
            
            m_W = beta1 * m_W + (1 - beta1) * gW
            v_W = beta2 * v_W + (1 - beta2) * gW**2
            m_W_hat = m_W / (1 - beta1**t)
            v_W_hat = v_W / (1 - beta2**t)
            
            U_curr -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            V_curr -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            W_curr -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
        # Try to round to half-integers
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if check_decomposition(U_rounded, V_rounded, W_rounded):
            return (U_rounded, V_rounded, W_rounded)
            
        # Also try rounding to integers
        U_int = np.round(best_U)
        V_int = np.round(best_V)
        W_int = np.round(best_W)
        
        if check_decomposition(U_int, V_int, W_int):
            return (U_int, V_int, W_int)
            
        return None
    
    def try_scipy_lbfgs(r, max_iters=100, seed=42):
        """Try to find a rank-r decomposition using scipy L-BFGS-B."""
        if time.time() > deadline:
            return None
            
        np.random.seed(seed)
        
        init_U = np.random.randn(nm, r) * 0.5
        init_V = np.random.randn(mp, r) * 0.5
        init_W = np.random.randn(pn, r) * 0.5
        
        def unpack(params):
            U = params[:r*nm].reshape(nm, r)
            V = params[r*nm:r*(nm+mp)].reshape(mp, r)
            W = params[r*(nm+mp):].reshape(pn, r)
            return U, V, W
            
        def objective(params):
            U, V, W = unpack(params)
            T_rec = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            error = T_rec - T
            return np.sum(error**2)
            
        def gradient(params):
            U, V, W = unpack(params)
            T_rec = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            error = T_rec - T
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r_idx in range(r):
                grad_U[:, r_idx] = 2 * np.einsum('abc,b,c->a', error, V[:, r_idx], W[:, r_idx])
                grad_V[:, r_idx] = 2 * np.einsum('abc,a,c->b', error, U[:, r_idx], W[:, r_idx])
                grad_W[:, r_idx] = 2 * np.einsum('abc,a,b->c', error, U[:, r_idx], V[:, r_idx])
                
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
        init_params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        try:
            res = minimize(objective, init_params, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            return None
            
        if res.success or res.fun < 1e-8:
            U, V, W = unpack(res.x)
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if check_decomposition(U_rounded, V_rounded, W_rounded):
                return (U_rounded, V_rounded, W_rounded)
                
            # Try rounding to integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            if check_decomposition(U_int, V_int, W_int):
                return (U_int, V_int, W_int)
                
        return None
    
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try ranks from 50 down to 55
    ranks_to_try = [55, 54, 53, 52, 51, 50]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try multiple seeds and methods
        for seed in [42, 123, 456]:
            if time.time() > deadline:
                break
                
            # Try gradient descent
            result = try_gradient_descent(r, max_iters=150, seed=seed)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_result = (U, V, W)
                break
                
            # Try scipy L-BFGS-B
            result = try_scipy_lbfgs(r, max_iters=80, seed=seed)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_result = (U, V, W)
                break
                
        if best_rank < rank_sb:
            break
            
    U, V, W = best_result
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
