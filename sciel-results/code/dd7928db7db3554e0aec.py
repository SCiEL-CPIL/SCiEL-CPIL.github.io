import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def compute_residual(params, n, m, p, rank):
    """Compute the residual of the tensor decomposition."""
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U = params[:size_u].reshape((n * m, rank))
    V = params[size_u:size_u + size_v].reshape((m * p, rank))
    W = params[size_u + size_v:].reshape((p * n, rank))
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Target: T[i,j,k] = 1 for all i,j,k
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                residual += (val - 1.0) ** 2
    return residual

def compute_grad(params, n, m, p, rank):
    """Compute the gradient of the residual."""
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U = params[:size_u].reshape((n * m, rank))
    V = params[size_u:size_u + size_v].reshape((m * p, rank))
    W = params[size_u + size_v:].reshape((p * n, rank))
    
    # Precompute errors E[i,j,k]
    E = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                E[i, j, k] = val - 1.0
                
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                e = E[i, j, k]
                if e == 0.0: continue
                for r in range(rank):
                    # dF/dU[i,j,r] += 2 * E[i,j,k] * V[j,k,r] * W[k,i,r]
                    grad_U[i * m + j, r] += 2 * e * V[j * p + k, r] * W[k * n + i, r]
                    # dF/dV[j,k,r] += 2 * E[i,j,k] * U[i,j,r] * W[k,i,r]
                    grad_V[j * p + k, r] += 2 * e * U[i * m + j, r] * W[k * n + i, r]
                    # dF/dW[k,i,r] += 2 * E[i,j,k] * U[i,j,r] * V[j,k,r]
                    grad_W[k * n + i, r] += 2 * e * U[i * m + j, r] * V[j * p + k, r]
                    
    return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Start with schoolbook rank 45
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed))
    V_seed = np.zeros((m * p, rank_seed))
    W_seed = np.zeros((p * n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_seed
    best_U = U_seed.copy()
    best_V = V_seed.copy()
    best_W = W_seed.copy()
    
    # Try greedy removal with local optimization
    current_rank = rank_seed
    U = U_seed.copy()
    V = V_seed.copy()
    W = W_seed.copy()
    
    def check_valid(U, V, W, rank, tol=1e-6):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Greedy removal: try removing each term and see if we can re-optimize to validity
    for _ in range(30):  # Max 30 removal attempts (increased from 20 due to speed)
        if time.time() > deadline:
            break
        
        removed = False
        for r in range(current_rank):
            if time.time() > deadline:
                break
            
            # Try removing term r
            mask = [x for x in range(current_rank) if x != r]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            new_rank = current_rank - 1
            
            # Check if already valid
            if check_valid(U_new, V_new, W_new, new_rank):
                U, V, W = U_new, V_new, W_new
                current_rank = new_rank
                removed = True
                break
            
            # Try local optimization to recover validity
            # Flatten
            size_u = n * m * new_rank
            size_v = m * p * new_rank
            size_w = p * n * new_rank
            x0 = np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()])
            
            # Quick optimization with limited iterations
            try:
                # Use BFGS with analytic gradient for speed and accuracy
                result = minimize(
                    compute_residual, x0, args=(n, m, p, new_rank),
                    jac=compute_grad,
                    method='BFGS',
                    options={'maxiter': 100, 'gtol': 1e-6, 'disp': False}
                )
                if result.fun < 1e-8:
                    # Check validity
                    U_opt = result.x[:size_u].reshape((n * m, new_rank))
                    V_opt = result.x[size_u:size_u + size_v].reshape((m * p, new_rank))
                    W_opt = result.x[size_u + size_v:].reshape((p * n, new_rank))
                    if check_valid(U_opt, V_opt, W_opt, new_rank, tol=1e-5):
                        U, V, W = U_opt, V_opt, W_opt
                        current_rank = new_rank
                        removed = True
                        break
            except Exception:
                pass
        
        if not removed:
            break
    
    best_rank = current_rank
    best_U = U
    best_V = V
    best_W = W
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
