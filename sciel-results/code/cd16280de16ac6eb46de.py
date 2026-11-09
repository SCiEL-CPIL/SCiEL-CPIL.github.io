import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def _build_target_tensor(n, m, p):
    # T[i, j, k, l] = 1 if j==k else 0, for (A[i,j]*B[j,k]) -> C[i,k]
    # We want: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor M[i, j, k, l] such that C[i,l] = sum_{j,k} A[i,j] B[k,l] M[i,j,k,l]
    # M[i,j,k,l] = delta_{j,k}
    T = np.zeros((n, m, p, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k, k] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Product: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal delta_{j,k}
    r = U.shape[1]
    # Reshape
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # Compute T_hat[i, j, k, l] = sum_c U3[i,j,c] V3[j,k,c] W3[k,l,c]
    # Use einsum
    T_hat = np.einsum('ijc,jkc,kic->ijkl', U3, V3, W3)
    T_true = np.zeros((n, m, p, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i, j, k, k] = 1.0
    err = np.max(np.abs(T_hat - T_true))
    return err < tol

def _try_rank(r, n, m, p, n_trials=50, seed=0):
    """Try to find a rank-r decomposition via random init + Newton polish."""
    T_true = _build_target_tensor(n, m, p)
    rng = np.random.RandomState(seed)
    
    for trial in range(n_trials):
        # Random initialization
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Newton iterations on the bilinear system
        # f_c = sum_{i,j,k,l} T_true[i,j,k,l] * (product of U,V,W entries) ... 
        # Actually, the constraint is: for each (i,j,k,l): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+l,c] = T_true[i,j,k,l]
        # This is a system of n*m*p*p equations in r*(n*m + m*p + p*n) unknowns.
        # We use a Gauss-Newton / Levenberg-Marquardt style approach.
        
        for it in range(8):
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            
            T_hat = np.einsum('ijc,jkc,kic->ijkl', U3, V3, W3)
            residual = T_hat - T_true
            res_flat = residual.ravel()
            
            if np.max(np.abs(residual)) < 1e-10:
                return U, V, W
            
            # Build Jacobian (approximate via finite differences or analytic)
            # Analytic: d(T_hat[i,j,k,l])/dU[i2,j2,c] = delta_{i,i2} delta_{j,j2} * V[j,k,c]*W[k,l,c]
            # This is large; use a simpler approach: solve linear system per variable group.
            
            # Use a simple gradient step with adaptive learning rate
            # Gradient w.r.t. U[i,j,c] = sum_{k,l} residual[i,j,k,l] * V[j,k,c] * W[k,l,c]
            grad_U = np.einsum('ijkl,jkc,kic->ijc', residual, V3, W3)
            grad_V = np.einsum('ijkl,ijc,kic->jk c'.replace(' ',''), residual, U3, W3) if False else \
                     np.einsum('ijkl,ijc,kic->jkc', residual, U3, W3)
            grad_W = np.einsum('ijkl,ijc,jkc->kic', residual, U3, V3)
            
            grad_U_flat = grad_U.reshape(-1)
            grad_V_flat = grad_V.reshape(-1)
            grad_W_flat = grad_W.reshape(-1)
            
            all_grad = np.concatenate([grad_U_flat, grad_V_flat, grad_W_flat])
            gnorm = np.linalg.norm(all_grad)
            if gnorm < 1e-14:
                break
            
            # Adaptive step
            lr = 0.05 / (gnorm + 1e-8)
            # Try a few step sizes
            best_lr = lr
            best_err = np.max(np.abs(residual))
            for factor in [1.0, 0.5, 0.25, 0.1]:
                U_new = U - factor * lr * grad_U_flat.reshape(n, m, r).reshape(-1)
                V_new = V - factor * lr * grad_V_flat.reshape(m, p, r).reshape(-1)
                W_new = W - factor * lr * grad_W_flat.reshape(p, n, r).reshape(-1)
                U3n = U_new.reshape(n, m, r)
                V3n = V_new.reshape(m, p, r)
                W3n = W_new.reshape(p, n, r)
                Thn = np.einsum('ijc,jkc,kic->ijkl', U3n, V3n, W3n)
                errn = np.max(np.abs(Thn - T_true))
                if errn < best_err:
                    best_err = errn
                    best_lr = factor * lr
                    U, V, W = U_new, V_new, W_new
                if best_err < 1e-10:
                    return U, V, W
        
        # Final check
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Try ranks from 35 down to 1
    for r in range(35, 0, -1):
        if time.time() > deadline:
            break
        # Number of trials based on remaining time
        remaining = deadline - time.time()
        n_trials = max(5, min(200, int(remaining / 0.5)))
        
        result = _try_rank(r, n, m, p, n_trials=n_trials, seed=r * 1000)
        if result is not None:
            U, V, W = result
            # Convert to list of lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
