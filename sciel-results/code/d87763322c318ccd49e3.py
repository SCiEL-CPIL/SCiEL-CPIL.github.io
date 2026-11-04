import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U, V, W, best_rank = schoolbook()
    
    def check_solution(U, V, W, tol=1e-6):
        """Check if the tensor decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                for b in range(m):
                    for c in range(m):
                        val = 0.0
                        for r in range(rank):
                            val += W[k*n+i, r] * U[i*m+b, r] * V[c*p+k, r]
                        expected = 1.0 if b == c else 0.0
                        if abs(val - expected) > tol:
                            return False
        return True
    
    def to_half_int(arr):
        return np.round(arr * 2) / 2
    
    def objective_simple(x, rank):
        n, m, p = DIMENSIONS
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank+m*p*rank:].reshape(p*n, rank)
        
        U_reshaped = U.reshape(n, m, rank)  # U[i, b, r]
        V_reshaped = V.reshape(m, p, rank)  # V[c, k, r]
        W_reshaped = W.reshape(p, n, rank)  # W[k, i, r]
        
        # T[i, b, c, k] = sum_r W[k,i,r] * U[i,b,r] * V[c,k,r]
        T = np.einsum('kir,ibr,ckr->ibck', W_reshaped, U_reshaped, V_reshaped)
        
        # Residuals: T[i,b,c,k] - delta(b,c)
        residuals = np.zeros((n, m, m, p))
        for i in range(n):
            for k in range(p):
                for b in range(m):
                    for c in range(m):
                        residuals[i, b, c, k] = T[i, b, c, k] - (1.0 if b == c else 0.0)
        
        return np.sum(residuals ** 2)
    
    def gradient_analytic(x, rank):
        """Compute analytic gradient of objective."""
        n, m, p = DIMENSIONS
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank+m*p*rank:].reshape(p*n, rank)
        
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # T[i,b,c,k] = sum_r W[k,i,r] * U[i,b,r] * V[c,k,r]
        # dT/dU[i,b,r] = W[k,i,r] * V[c,k,r] summed over k
        # dT/dV[c,k,r] = W[k,i,r] * U[i,b,r] summed over i
        # dT/dW[k,i,r] = U[i,b,r] * V[c,k,r]
        
        # Residual R[i,b,c,k] = T[i,b,c,k] - delta(b,c)
        # Objective = sum_{i,b,c,k} R[i,b,c,k]^2
        # dObj/dU[i,b,r] = sum_{i,b,c,k} 2*R[i,b,c,k] * dT[i,b,c,k]/dU[i,b,r]
        # = sum_{c,k} 2*R[i,b,c,k] * W[k,i,r] * V[c,k,r]
        
        # Compute R
        T = np.einsum('kir,ibr,ckr->ibck', W_reshaped, U_reshaped, V_reshaped)
        R = np.zeros((n, m, m, p))
        for i in range(n):
            for k in range(p):
                for b in range(m):
                    for c in range(m):
                        R[i, b, c, k] = T[i, b, c, k] - (1.0 if b == c else 0.0)
        
        # Gradient for U: dU[i,b,r] = sum_{c,k} 2*R[i,b,c,k] * W[k,i,r] * V[c,k,r]
        grad_U = np.zeros((n, m, rank))
        for i in range(n):
            for b in range(m):
                for r in range(rank):
                    grad_U[i, b, r] = 2.0 * np.sum(
                        R[i, b, :, :] * W_reshaped[:, i, r] * V_reshaped[:, :, r]
                    )
        
        # Gradient for V: dV[c,k,r] = sum_{i,b} 2*R[i,b,c,k] * W[k,i,r] * U[i,b,r]
        grad_V = np.zeros((m, p, rank))
        for c in range(m):
            for k in range(p):
                for r in range(rank):
                    grad_V[c, k, r] = 2.0 * np.sum(
                        R[:, :, c, k] * W_reshaped[k, :, r] * U_reshaped[:, :, r]
                    )
        
        # Gradient for W: dW[k,i,r] = sum_{b,c} 2*R[i,b,c,k] * U[i,b,r] * V[c,k,r]
        grad_W = np.zeros((p, n, rank))
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    grad_W[k, i, r] = 2.0 * np.sum(
                        R[i, :, :, k] * U_reshaped[i, :, r] * V_reshaped[:, k, r]
                    )
        
        grad = np.concatenate([
            grad_U.flatten(),
            grad_V.flatten(),
            grad_W.flatten()
        ])
        return grad
    
    # Try ranks from lower bound up
    lower_bound = max(n*p, m*p, n*m)
    
    for target_rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 25:
            break
        
        n_params = (n*m + m*p + p*n) * target_rank
        
        for attempt in range(40):
            if time.time() > deadline - 15:
                break
            
            # Initialize with various strategies
            np.random.seed(attempt * 7919 + target_rank * 104729)
            
            if attempt % 3 == 0:
                # Random Gaussian
                U_init = np.random.randn(n*m, target_rank) * 0.3
                V_init = np.random.randn(m*p, target_rank) * 0.3
                W_init = np.random.randn(p*n, target_rank) * 0.3
            elif attempt % 3 == 1:
                # Sparse initialization
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    # Pick random positions
                    u_idx = np.random.choice(n*m, size=min(3, n*m), replace=False)
                    v_idx = np.random.choice(m*p, size=min(3, m*p), replace=False)
                    w_idx = np.random.choice(p*n, size=min(3, p*n), replace=False)
                    U_init[u_idx, r] = np.random.choice([-1, 1], size=len(u_idx))
                    V_init[v_idx, r] = np.random.choice([-1, 1], size=len(v_idx))
                    W_init[w_idx, r] = np.random.choice([-1, 1], size=len(w_idx))
            else:
                # Structured: try to match schoolbook pattern partially
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    i = r % n
                    b = (r // n) % m
                    k = (r // (n*m)) % p
                    U_init[i*m+b, r] = 1.0
                    V_init[b*p+k, r] = 1.0
                    W_init[k*n+i, r] = 1.0
                    # Add small noise
                    U_init[r % (n*m), r] += np.random.randn() * 0.1
                    V_init[r % (m*p), r] += np.random.randn() * 0.1
                    W_init[r % (p*n), r] += np.random.randn() * 0.1
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Try L-BFGS-B with analytic gradient
            try:
                result = minimize(objective_simple, x0, args=(target_rank,), method='L-BFGS-B',
                                  jac=gradient_analytic, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
                
                if result.fun < 1e-10:
                    x_opt = result.x
                    U_opt = x_opt[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = x_opt[n*m*target_rank:n*m*rank+m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = x_opt[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Round to half-integers
                    U_h = to_half_int(U_opt)
                    V_h = to_half_int(V_opt)
                    W_h = to_half_int(W_opt)
                    
                    if check_solution(U_h, V_h, W_h, tol=1e-6):
                        best_rank = target_rank
                        U, V, W = U_h, V_h, W_h
                        break
                    else:
                        # Try without rounding
                        if check_solution(U_opt, V_opt, W_opt, tol=1e-6):
                            best_rank = target_rank
                            U, V, W = U_opt, V_opt, W_opt
                            break
            except Exception:
                pass
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
