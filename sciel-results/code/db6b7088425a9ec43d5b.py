import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Schoolbook decomposition
    rank_school = n * m * p  # 45
    U_fb = np.zeros((n*m, rank_school), dtype=np.float64)
    V_fb = np.zeros((m*p, rank_school), dtype=np.float64)
    W_fb = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1

    target = np.ones((n, m, p), dtype=np.float64)

    def verify(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(result, target, atol=1e-9, rtol=1e-9)

    def compute_error_and_grad(params, R):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = params[:size_u*R].reshape(size_u, R)
        V = params[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
        W = params[size_u*R + size_v*R :].reshape(size_w, R)
        
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        diff = result - target
        error = np.sum(diff**2)
        
        # Compute gradients
        # d(error)/d(U_r[i,j,c]) = 2 * sum_{k} diff[i,j,k] * V_r[j,k,c] * W_r[k,i,c]
        # d(error)/d(V_r[j,k,c]) = 2 * sum_{i} diff[i,j,k] * U_r[i,j,c] * W_r[k,i,c]
        # d(error)/d(W_r[k,i,c]) = 2 * sum_{j} diff[i,j,k] * U_r[i,j,c] * V_r[j,k,c]
        
        grad_U_r = 2 * np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
        grad_V_r = 2 * np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
        grad_W_r = 2 * np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
        
        grad_U = grad_U_r.reshape(size_u, R)
        grad_V = grad_V_r.reshape(size_v, R)
        grad_W = grad_W_r.reshape(size_w, R)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return error, grad

    def gradient_descent(params0, R, max_iters=500, lr=0.01, tol=1e-10):
        params = params0.copy()
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            error, grad = compute_error_and_grad(params, R)
            
            if error < tol:
                break
            
            # Adaptive learning rate
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-12:
                break
            
            params = params - lr * grad
            
            # Simple momentum-like approach: reduce lr if error increases
            # (simplified: just keep going)
        
        error, _ = compute_error_and_grad(params, R)
        if error < 1e-8:
            size_u = n * m
            size_v = m * p
            size_w = p * n
            U = params[:size_u*R].reshape(size_u, R)
            V = params[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
            W = params[size_u*R + size_v*R :].reshape(size_w, R)
            
            # Snap to nearest 0.5
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if verify(U_snap, V_snap, W_snap):
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
        
        return None

    def solve_rank(R, max_restarts=200):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = (size_u + size_v + size_w) * R
        
        best_result = None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Multiple initialization strategies
            if restart % 4 == 0:
                # Random small integers
                params0 = np.random.randint(-2, 3, size=total_size).astype(np.float64)
            elif restart % 4 == 1:
                # Random continuous
                params0 = np.random.randn(total_size).astype(np.float64) * 0.5
            elif restart % 4 == 2:
                # Structured: start from schoolbook and perturb
                if R <= rank_school:
                    # Take first R columns of schoolbook
                    U_init = U_fb[:, :R].copy()
                    V_init = V_fb[:, :R].copy()
                    W_init = W_fb[:, :R].copy()
                    # Add small random perturbation
                    U_init += np.random.randn(*U_init.shape) * 0.1
                    V_init += np.random.randn(*V_init.shape) * 0.1
                    W_init += np.random.randn(*W_init.shape) * 0.1
                    params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                else:
                    params0 = np.random.randn(total_size).astype(np.float64) * 0.5
            else:
                # Binary-like initialization
                params0 = np.random.choice([-1.0, 0.0, 1.0], size=total_size)
            
            result = gradient_descent(params0, R, max_iters=300, lr=0.02)
            if result is not None:
                return result
        
        return best_result

    # Try to find lower ranks
    # Start from 44 down to 20
    # Limit total time spent on search
    for R in range(44, 19, -1):
        if time.time() > deadline - 15:
            break
        
        result = solve_rank(R, max_restarts=100)
        if result is not None:
            return result

    # Return schoolbook
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
