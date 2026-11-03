"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # Schoolbook fallback
    rank_school = n * m * p  # 54
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
    best_decomp = (U_school, V_school, W_school)
    best_rank = rank_school

    def verify_decomp_336(U, V, W, tol=1e-8):
        """Verify that U, V, W decompose the 3x3x6 multiplication tensor."""
        n, m, p = 3, 3, 6
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for r_idx in range(r):
            u = U[:, r_idx].reshape(n, m)
            v = V[:, r_idx].reshape(m, p)
            w = W[:, r_idx].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, 1.0, atol=tol)

    def verify_decomp_333(U, V, W, tol=1e-8):
        """Verify that U, V, W decompose the 3x3x3 multiplication tensor."""
        n, m, p = 3, 3, 3
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for r_idx in range(r):
            u = U[:, r_idx].reshape(n, m)
            v = V[:, r_idx].reshape(m, p)
            w = W[:, r_idx].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, 1.0, atol=tol)

    def extend_333_to_336(U3, V3, W3):
        """Extend a 3x3x3 decomposition to 3x3x6 by treating B as two 3x3 blocks."""
        r = U3.shape[1]
        rank = 2 * r
        U = np.zeros((9, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        # First half: B1 (columns 0-2)
        U[:, :r] = U3
        for j in range(3):
            for k in range(3):
                V[j * 6 + k, :r] = V3[j * 3 + k, :]
        for k in range(3):
            for i in range(3):
                W[k * 3 + i, :r] = W3[k * 3 + i, :]
        
        # Second half: B2 (columns 3-5)
        U[:, r:] = U3
        for j in range(3):
            for k in range(3):
                V[j * 6 + (k + 3), r:] = V3[j * 3 + k, :]
        for k in range(3):
            for i in range(3):
                W[(k + 3) * 3 + i, r:] = W3[k * 3 + i, :]
        
        return U, V, W

    def compute_tensor_333(U, V, W):
        """Compute the tensor from a 3x3x3 decomposition."""
        n, m, p = 3, 3, 3
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for r_idx in range(r):
            u = U[:, r_idx].reshape(n, m)
            v = V[:, r_idx].reshape(m, p)
            w = W[:, r_idx].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return T

    def compute_tensor_336(U, V, W):
        """Compute the tensor from a 3x3x6 decomposition."""
        n, m, p = 3, 3, 6
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for r_idx in range(r):
            u = U[:, r_idx].reshape(n, m)
            v = V[:, r_idx].reshape(m, p)
            w = W[:, r_idx].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return T

    def loss_333(params, r):
        """Loss function for 3x3x3 rank-r decomposition."""
        n, m, p = 3, 3, 3
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r :].reshape(p*n, r)
        T = compute_tensor_333(U, V, W)
        err = T - 1.0
        return np.sum(err * err)

    def polish_333(U, V, W, max_iter=200, tol=1e-10):
        """Polish a 3x3x3 decomposition using Newton-like steps."""
        n, m, p = 3, 3, 3
        r = U.shape[1]
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            T = compute_tensor_333(
                params[:n*m*r].reshape(n*m, r),
                params[n*m*r : n*m*r + m*p*r].reshape(m*p, r),
                params[n*m*r + m*p*r :].reshape(p*n, r)
            )
            err = T - 1.0
            loss = np.sum(err * err)
            if loss < tol:
                break
            
            # Compute gradients via finite differences or analytic
            # Use simple gradient descent with adaptive step
            U = params[:n*m*r].reshape(n*m, r)
            V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W = params[n*m*r + m*p*r :].reshape(p*n, r)
            
            # Gradient with respect to U
            grad_U = np.zeros_like(U)
            for r_idx in range(r):
                v = V[:, r_idx].reshape(m, p)
                w = W[:, r_idx].reshape(p, n)
                # d/dU[i,j,r_idx] of sum err^2
                # err[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1
                # d(err^2)/dU[i,j,r_idx] = 2*err[i,j,k]*V[j,k,r_idx]*W[k,i,r_idx]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            grad_U[i*m+j, r_idx] += 2 * err[i,j,k] * v[j,k] * w[k,i]
            
            # Gradient with respect to V
            grad_V = np.zeros_like(V)
            for r_idx in range(r):
                u = U[:, r_idx].reshape(n, m)
                w = W[:, r_idx].reshape(p, n)
                for j in range(m):
                    for k in range(p):
                        for i in range(n):
                            grad_V[j*p+k, r_idx] += 2 * err[i,j,k] * u[i,j] * w[k,i]
            
            # Gradient with respect to W
            grad_W = np.zeros_like(W)
            for r_idx in range(r):
                u = U[:, r_idx].reshape(n, m)
                v = V[:, r_idx].reshape(m, p)
                for k in range(p):
                    for i in range(n):
                        for j in range(m):
                            grad_W[k*n+i, r_idx] += 2 * err[i,j,k] * u[i,j] * v[j,k]
            
            grads = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            grad_norm = np.linalg.norm(grads)
            if grad_norm < 1e-12:
                break
            
            # Adaptive step size
            step = 0.01
            for _ in range(10):
                new_params = params - step * grads
                new_U = new_params[:n*m*r].reshape(n*m, r)
                new_V = new_params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                new_W = new_params[n*m*r + m*p*r :].reshape(p*n, r)
                new_T = compute_tensor_333(new_U, new_V, new_W)
                new_loss = np.sum((new_T - 1.0)**2)
                if new_loss < loss:
                    params = new_params
                    break
                step *= 0.5
            
            if step < 1e-10:
                break
        
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r :].reshape(p*n, r)
        return U, V, W

    def try_rank_23_333():
        """Try to find a rank-23 decomposition of 3x3x3."""
        if time.time() > deadline - 5.0:
            return None
        
        n, m, p = 3, 3, 3
        r = 23
        
        # Use structured initialization based on known algebraic patterns
        # The rank-23 decomposition has a specific structure with entries in {-1, 0, 1, 1/2}
        
        # Try multiple random restarts with gradient-based optimization
        best_U, best_V, best_W = None, None, None
        best_loss = float('inf')
        
        num_trials = 8
        for trial in range(num_trials):
            if time.time() > deadline - 5.0:
                break
            
            np.random.seed(42 + trial * 137)
            
            # Initialize with small random values
            U = np.random.randn(n*m, r) * 0.3
            V = np.random.randn(m*p, r) * 0.3
            W = np.random.randn(p*n, r) * 0.3
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B
            import scipy.optimize as optimize
            
            def loss_fn(params):
                U = params[:n*m*r].reshape(n*m, r)
                V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W = params[n*m*r + m*p*r :].reshape(p*n, r)
                T = compute_tensor_333(U, V, W)
                err = T - 1.0
                return np.sum(err * err)
            
            try:
                result = optimize.minimize(
                    loss_fn, params, method='L-BFGS-B',
                    maxiter=500, options={'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
            
            U_opt = result.x[:n*m*r].reshape(n*m, r)
            V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_U, best_V, best_W = U_opt, V_opt, W_opt
            
            if best_loss < 1e-8:
                break
        
        if best_loss < 1e-6:
            # Polish
            U_p, V_p, W_p = polish_333(best_U, best_V, W_opt, max_iter=100, tol=1e-12)
            T = compute_tensor_333(U_p, V_p, W_p)
            final_loss = np.sum((T - 1.0)**2)
            
            if final_loss < 1e-8:
                # Round to nearest 0.5
                U_rounded = np.round(U_p * 2) / 2
                V_rounded = np.round(V_p * 2) / 2
                W_rounded = np.round(W_p * 2) / 2
                
                if verify_decomp_333(U_rounded, V_rounded, W_rounded, tol=1e-6):
                    return U_rounded, V_rounded, W_rounded
                else:
                    # Try rounding to integers
                    U_rounded = np.round(U_p)
                    V_rounded = np.round(V_p)
                    W_rounded = np.round(W_p)
                    if verify_decomp_333(U_rounded, V_rounded, W_rounded, tol=1e-6):
                        return U_rounded, V_rounded, W_rounded
                    else:
                        # Return unrounded if close enough
                        return U_p, V_p, W_p
        
        return None

    # Try to find rank-23 decomposition of 3x3x3
    if time.time() < deadline - 10.0:
        result_23 = try_rank_23_333()
        if result_23 is not None:
            U3, V3, W3 = result_23
            U336, V336, W336 = extend_333_to_336(U3, V3, W3)
            if verify_decomp_336(U336, V336, W336, tol=1e-6):
                best_rank = 46
                best_decomp = (U336, V336, W336)
                return {"U": U336.tolist(), "V": V336.tolist(), "W": W336.tolist()}
            else:
                # Try polishing the 3x3x6 directly
                r = U336.shape[1]
                n, m, p = 3, 3, 6
                params = np.concatenate([U336.flatten(), V336.flatten(), W336.flatten()])
                
                # Simple polish on 3x3x6
                for it in range(50):
                    if time.time() > deadline:
                        break
                    U = params[:n*m*r].reshape(n*m, r)
                    V = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W = params[n*m*r + m*p*r :].reshape(p*n, r)
                    T = compute_tensor_336(U, V, W)
                    err = T - 1.0
                    loss = np.sum(err * err)
                    if loss < 1e-10:
                        break
                    
                    # Compute gradients
                    grad_U = np.zeros_like(U)
                    grad_V = np.zeros_like(V)
                    grad_W = np.zeros_like(W)
                    
                    for r_idx in range(r):
                        u = U[:, r_idx].reshape(n, m)
                        v = V[:, r_idx].reshape(m, p)
                        w = W[:, r_idx].reshape(p, n)
                        
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    grad_U[i*m+j, r_idx] += 2 * err[i,j,k] * v[j,k] * w[k,i]
                                    grad_V[j*p+k, r_idx] += 2 * err[i,j,k] * u[i,j] * w[k,i]
                                    grad_W[k*n+i, r_idx] += 2 * err[i,j,k] * u[i,j] * v[j,k]
                    
                    grads = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                    grad_norm = np.linalg.norm(grads)
                    if grad_norm < 1e-12:
                        break
                    
                    step = 0.01
                    for _ in range(10):
                        new_params = params - step * grads
                        new_U = new_params[:n*m*r].reshape(n*m, r)
                        new_V = new_params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                        new_W = new_params[n*m*r + m*p*r :].reshape(p*n, r)
                        new_T = compute_tensor_336(new_U, new_V, new_W)
                        new_loss = np.sum((new_T - 1.0)**2)
                        if new_loss < loss:
                            params = new_params
                            break
                        step *= 0.5
                    
                    if step < 1e-10:
                        break
                
                U_final = params[:n*m*r].reshape(n*m, r)
                V_final = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_final = params[n*m*r + m*p*r :].reshape(p*n, r)
                
                T_final = compute_tensor_336(U_final, V_final, W_final)
                if np.allclose(T_final, 1.0, atol=1e-6):
                    best_rank = 46
                    best_decomp = (U_final, V_final, W_final)
                    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    # Fall back to schoolbook
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
