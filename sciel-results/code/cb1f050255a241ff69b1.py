import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and i == i2 and k == k2:
                        expected = 1.0
                    else:
                        expected = 0.0
                    if abs(val - expected) > tol:
                        return False
        return True
    
    best_result = schoolbook()
    
    # Try to find a rank-28 solution using a more efficient approach
    rank = 28
    if time.time() < deadline - 30:
        # Precompute the active and inactive triples
        active = []
        inactive = []
        for a in range(n*m):
            i = a // m
            j = a % m
            for b in range(m*p):
                j2 = b // p
                k = b % p
                for c in range(p*n):
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and i == i2 and k == k2:
                        active.append((a, b, c))
                    else:
                        inactive.append((a, b, c))
        
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed)
            
            # Initialize with small random values
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective_and_grad(params):
                U_flat = params[:n*m*rank].reshape(n*m, rank)
                V_flat = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W_flat = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
                
                loss = 0.0
                grad_U = np.zeros_like(U_flat)
                grad_V = np.zeros_like(V_flat)
                grad_W = np.zeros_like(W_flat)
                
                # Active equations
                for a, b, c in active:
                    val = np.sum(U_flat[a] * V_flat[b] * W_flat[c])
                    err = val - 1.0
                    loss += err**2
                    # Gradient
                    for t in range(rank):
                        uvw = U_flat[a, t] * V_flat[b, t] * W_flat[c, t]
                        grad_U[a, t] += 2 * err * V_flat[b, t] * W_flat[c, t]
                        grad_V[b, t] += 2 * err * U_flat[a, t] * W_flat[c, t]
                        grad_W[c, t] += 2 * err * U_flat[a, t] * V_flat[b, t]
                
                # Inactive equations (subset for speed)
                for a, b, c in inactive[:100]:  # Only check 100 inactive equations
                    val = np.sum(U_flat[a] * V_flat[b] * W_flat[c])
                    loss += val**2
                    for t in range(rank):
                        grad_U[a, t] += 2 * val * V_flat[b, t] * W_flat[c, t]
                        grad_V[b, t] += 2 * val * U_flat[a, t] * W_flat[c, t]
                        grad_W[c, t] += 2 * val * U_flat[a, t] * V_flat[b, t]
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                return loss, grad
            
            try:
                result = minimize(objective_and_grad, params0, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10})
                
                if result.fun < 1e-6:
                    U_sol = result.x[:n*m*rank].reshape(n*m, rank)
                    V_sol = result.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                    W_sol = result.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    
                    if verify(U_sol, V_sol, W_sol, tol=1e-7):
                        return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
            except Exception:
                pass
    
    return best_result
