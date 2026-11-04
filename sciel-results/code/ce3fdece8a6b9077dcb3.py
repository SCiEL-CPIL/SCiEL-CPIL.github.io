import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_49(n, m, p):
    """
    Attempt to find rank-49 decomposition for 2x5x5 matrix multiplication.
    Uses a targeted approach: start from schoolbook and try to merge two terms.
    """
    if time.time() > _DEADLINE - 20:
        return None

    target_rank = 49
    
    # Start from schoolbook decomposition
    schoolbook = _schoolbook(n, m, p)
    U_sb = np.array(schoolbook['U'], dtype=float)
    V_sb = np.array(schoolbook['V'], dtype=float)
    W_sb = np.array(schoolbook['W'], dtype=float)
    
    # The schoolbook has 50 terms. We need to see if we can represent
    # 50 rank-1 tensors as 49 rank-1 tensors.
    # 
    # Key insight: if there's a linear dependency among the 50 schoolbook
    # terms, we might be able to express one as a combination of others.
    #
    # For matrix multiplication, the tensor is:
    # T_{i,j,k} = sum_l A_{i,l} B_{l,k}
    # 
    # The schoolbook decomposition uses terms for each (i,j,k) triple.
    # A rank-49 decomposition would need to "merge" two of these terms.
    #
    # Strategy: Try to find coefficients such that one schoolbook term
    # can be expressed as a linear combination of the other 49 terms.
    
    # For each pair of schoolbook indices to potentially merge,
    # try to find rational coefficients
    
    # First, let's try a simpler approach: use the fact that for 2x5x5,
    # there might be a known lower bound or construction.
    # 
    # Actually, let's try a numerical approach with better initialization.
    # Start from schoolbook and perturb slightly.
    
    best_error = float('inf')
    best_params = None
    
    # Try different strategies for merging
    strategies = [
        'perturb_schoolbook',
        'random_with_structure',
        'gradient_descent'
    ]
    
    for strategy in strategies:
        if time.time() > _DEADLINE - 5:
            break
        
        if strategy == 'perturb_schoolbook':
            # Start from schoolbook and try to zero out one column
            # while adjusting others
            for col_to_remove in range(50):
                if time.time() > _DEADLINE - 5:
                    break
                
                # Create initial guess: schoolbook with one column removed
                # and others adjusted
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                
                # Copy all columns except col_to_remove
                new_col = 0
                for old_col in range(50):
                    if old_col == col_to_remove:
                        continue
                    U_init[:, new_col] = U_sb[:, old_col]
                    V_init[:, new_col] = V_sb[:, old_col]
                    W_init[:, new_col] = W_sb[:, old_col]
                    new_col += 1
                
                # Now we need to add the contribution of col_to_remove
                # to the remaining 49 columns
                # This is a linear problem: find delta_U, delta_V, delta_W
                # such that the tensor equation still holds
                
                # For now, just try to optimize from this starting point
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                def objective(params):
                    U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, 2*n*m*target_rank])
                    U = U_flat.reshape(n*m, target_rank)
                    V = V_flat.reshape(m*p, target_rank)
                    W = W_flat.reshape(p*n, target_rank)
                    
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(target_rank):
                                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                error += (s - 1.0)**2
                    return error
                
                try:
                    import scipy.optimize as opt
                    res = opt.minimize(objective, params, method='L-BFGS-B', 
                                       options={'maxiter': 100, 'ftol': 1e-10})
                    
                    if res.fun < best_error:
                        best_error = res.fun
                        best_params = res.x
                        
                except Exception:
                    pass
        
        elif strategy == 'random_with_structure':
            # Random initialization but with some structure
            for restart in range(5):
                if time.time() > _DEADLINE - 5:
                    break
                
                np.random.seed(100 + restart)
                
                # Initialize with small values
                U_init = np.random.randn(n*m, target_rank) * 0.1
                V_init = np.random.randn(m*p, target_rank) * 0.1
                W_init = np.random.randn(p*n, target_rank) * 0.1
                
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                def objective(params):
                    U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, 2*n*m*target_rank])
                    U = U_flat.reshape(n*m, target_rank)
                    V = V_flat.reshape(m*p, target_rank)
                    W = W_flat.reshape(p*n, target_rank)
                    
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(target_rank):
                                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                error += (s - 1.0)**2
                    return error
                
                try:
                    import scipy.optimize as opt
                    res = opt.minimize(objective, params, method='L-BFGS-B', 
                                       options={'maxiter': 100, 'ftol': 1e-10})
                    
                    if res.fun < best_error:
                        best_error = res.fun
                        best_params = res.x
                        
                except Exception:
                    pass
        
        elif strategy == 'gradient_descent':
            # Simple gradient descent with momentum
            if time.time() > _DEADLINE - 5:
                break
            
            np.random.seed(42)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            lr = 0.01
            momentum = 0.9
            vel_U = np.zeros_like(U)
            vel_V = np.zeros_like(V)
            vel_W = np.zeros_like(W)
            
            for step in range(200):
                if time.time() > _DEADLINE - 5:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err = s - 1.0
                            for c in range(target_rank):
                                grad_U[i*m+j, c] += err * V[j*p+k, c] * W[k*n+i, c]
                                grad_V[j*p+k, c] += err * U[i*m+j, c] * W[k*n+i, c]
                                grad_W[k*n+i, c] += err * U[i*m+j, c] * V[j*p+k, c]
                
                # Update with momentum
                vel_U = momentum * vel_U - lr * grad_U
                vel_V = momentum * vel_V - lr * grad_V
                vel_W = momentum * vel_W - lr * grad_W
                
                U += vel_U
                V += vel_V
                W += vel_W
                
                # Check error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            error += (s - 1.0)**2
                
                if error < best_error:
                    best_error = error
                    best_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        if best_error < 1e-6 and best_params is not None:
            # Round to rationals
            U_flat, V_flat, W_flat = np.split(best_params, [n*m*target_rank, 2*n*m*target_rank])
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Try multiple rounding schemes
            for denom in [1, 2, 4, 8, 16]:
                U_r = np.round(U * denom) / denom
                V_r = np.round(V * denom) / denom
                W_r = np.round(W * denom) / denom
                
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find rank 49
    result = _try_rank_49(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
