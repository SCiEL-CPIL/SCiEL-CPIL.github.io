import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    return tensor

def _compute_residuals(U, V, W, tensor, n, m, p):
    """Compute residuals for all entries."""
    rank = U.shape[1]
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                residuals.append(val - tensor[i*m+j, j*p+k, k*n+i])
    return np.array(residuals)

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is valid."""
    rank = U.shape[1]
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check invalid entries
    for i1 in range(n):
        for j1 in range(m):
            for k1 in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i1 == i2 and j1 == j2 and k1 == k2:
                                continue
                            val = np.sum(U[i1*m+j1, :] * V[j2*p+k1, :] * W[k2*n+i2, :])
                            if abs(val) > tol:
                                return False
    return True

def _try_rank(n, m, p, target_rank, max_iter=200, timeout=20):
    """Try to find a valid decomposition for the given target rank using gradient-based optimization."""
    start_time = time.time()
    
    n_params = target_rank * (n*m + m*p + p*n)
    
    def unpack_params(params):
        U = params[:target_rank*n*m].reshape(target_rank, n*m).T
        V = params[target_rank*n*m:target_rank*(n*m+m*p)].reshape(target_rank, m*p).T
        W = params[target_rank*(n*m+m*p):].reshape(target_rank, p*n).T
        return U, V, W
    
    def objective(params):
        U, V, W = unpack_params(params)
        residuals = []
        
        # All valid and invalid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    residuals.append(val - 1.0)
        
        # Invalid entries
        for i1 in range(n):
            for j1 in range(m):
                for k1 in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i1 == i2 and j1 == j2 and k1 == k2:
                                    continue
                                val = np.sum(U[i1*m+j1, :] * V[j2*p+k1, :] * W[k2*n+i2, :])
                                residuals.append(val)
        
        return np.sum(np.array(residuals) ** 2)
    
    def gradient(params):
        U, V, W = unpack_params(params)
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Gradient for valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = val - 1.0
                    grad_U[i*m+j, :] += 2 * err * V[j*p+k, :] * W[k*n+i, :]
                    grad_V[j*p+k, :] += 2 * err * U[i*m+j, :] * W[k*n+i, :]
                    grad_W[k*n+i, :] += 2 * err * U[i*m+j, :] * V[j*p+k, :]
        
        # Gradient for invalid entries
        for i1 in range(n):
            for j1 in range(m):
                for k1 in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i1 == i2 and j1 == j2 and k1 == k2:
                                    continue
                                val = np.sum(U[i1*m+j1, :] * V[j2*p+k1, :] * W[k2*n+i2, :])
                                grad_U[i1*m+j1, :] += 2 * val * V[j2*p+k1, :] * W[k2*n+i2, :]
                                grad_V[j2*p+k1, :] += 2 * val * U[i1*m+j1, :] * W[k2*n+i2, :]
                                grad_W[k2*n+i2, :] += 2 * val * U[i1*m+j1, :] * V[j2*p+k1, :]
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return grad
    
    best_params = None
    best_obj = np.inf
    
    for trial in range(20):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(trial * 7919 + 13)
        x0 = np.random.randn(n_params) * 0.3
        
        try:
            result = minimize(objective, x0, jac=gradient, method='BFGS', 
                             options={'maxiter': max_iter, 'gtol': 1e-8})
            final_obj = result.fun
            if final_obj < best_obj:
                best_obj = final_obj
                best_params = result.x
        except Exception:
            continue
    
    if best_params is None or best_obj > 1e-6:
        return None
    
    U, V, W = unpack_params(best_params)
    
    # Verify with a stricter check
    if _check_decomposition(U, V, W, n, m, p, tol=1e-6):
        # Try to round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if _check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
            return U_rounded, V_rounded, W_rounded
        else:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, c] = 1.0
                V_school[j * p + k, c] = 1.0
                W_school[k * n + i, c] = 1.0
                c += 1
    
    # Try to find lower rank
    # Lower bound: max(n*m, m*p, p*n) = 21
    for target_rank in range(22, 42):
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, target_rank, max_iter=100, timeout=15)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
