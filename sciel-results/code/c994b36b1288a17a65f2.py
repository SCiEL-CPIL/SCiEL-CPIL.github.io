import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 8]
DEADLINE_SECONDS = 115

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _create_target_tensor(n, m, p):
    """Create the sparse target tensor as a dense array for easy comparison."""
    # Dimensions: (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1
    return T

def _reconstruct(U, V, W):
    """Reconstruct the tensor from factors."""
    # U: (nm, r), V: (mp, r), W: (pn, r)
    # Result: (nm, mp, pn)
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # Using einsum
    return np.einsum('ar,br,cr->abc', U, V, W)

def _error(U, V, W, T):
    """Calculate squared error."""
    T_rec = _reconstruct(U, V, W)
    return np.sum((T_rec - T) ** 2)

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    T_rec = _reconstruct(U, V, W)
    T_target = _create_target_tensor(n, m, p)
    return np.allclose(T_rec, T_target, atol=tol)

def _try_rank(r, n, m, p, T, time_budget, rng):
    """Try to find a rank-r decomposition."""
    start = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize factors
    # Use random initialization
    U = rng.normal(size=(nm, r))
    V = rng.normal(size=(mp, r))
    W = rng.normal(size=(pn, r))
    
    # Normalize to avoid scaling issues
    # We can fix the scale of one factor, e.g., make W have unit norm columns?
    # Or just let the optimizer handle it.
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Simple gradient descent
    # Since we have time, we can do a few iterations.
    # However, scipy.optimize might be better.
    
    # Let's try a few restarts with gradient descent
    for restart in range(5):
        U = rng.normal(size=(nm, r))
        V = rng.normal(size=(mp, r))
        W = rng.normal(size=(pn, r))
        
        # Scale factors to keep numbers reasonable
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        # Use L-BFGS-B or similar
        # Flatten variables
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat = x[:nm*r].reshape(nm, r)
            V_flat = x[nm*r:nm*r+mp*r].reshape(mp, r)
            W_flat = x[nm*r+mp*r:].reshape(pn, r)
            return _error(U_flat, V_flat, W_flat, T)
        
        # We can't easily use gradient descent with simple loops for large tensors in Python without being slow.
        # But nm=12, mp=32, pn=24, r~90.
        # x size ~ 90 * (12+32+24) = 5400.
        # Objective evaluation: einsum of 12x90, 32x90, 24x90 -> 12x32x24.
        # This is fast in numpy.
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 100})
            x_opt = res.x
            U_opt = x_opt[:nm*r].reshape(nm, r)
            V_opt = x_opt[nm*r:nm*r+mp*r].reshape(mp, r)
            W_opt = x_opt[nm*r+mp*r:].reshape(pn, r)
            
            err = res.fun
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                
            if time.time() - start > time_budget:
                break
        except Exception:
            pass
            
    # Round to nearest integer/half-integer
    # The problem allows integer/half-integer.
    # Let's try rounding to nearest integer first.
    U_round = np.round(best_U)
    V_round = np.round(best_V)
    W_round = np.round(best_W)
    
    if _verify_exact(U_round, V_round, W_round, n, m, p):
        return U_round, V_round, W_round, 0.0
        
    # Try half-integers (multiply by 2, round, divide by 2)
    U_half = np.round(best_U * 2) / 2
    V_half = np.round(best_V * 2) / 2
    W_half = np.round(best_W * 2) / 2
    
    if _verify_exact(U_half, V_half, W_half, n, m, p):
        return U_half, V_half, W_half, 0.0
        
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    T = _create_target_tensor(n, m, p)
    
    # Baseline: Schoolbook
    U_sb, V_sb, W_sb = _schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    rng = np.random.default_rng(42)
    
    # Try to improve
    # We can try ranks from best_rank - 1 down to some limit
    # But optimization is expensive.
    # Let's try a few specific ranks.
    
    # Known: Rank of M_{2,2,2} is 7.
    # Can we use this?
    # 3x4x8 can be embedded.
    
    # Let's just try to optimize for rank 95, 94, etc.
    # But 120s is short.
    # Let's try rank 90 as a test.
    
    # Actually, let's just return the schoolbook solution if we don't find better.
    # The goal is to beat 1.0.
    # If we find rank 95, score is 96/95 > 1.0.
    
    # Try a few ranks
    for r in range(best_rank - 1, max(best_rank - 10, 0), -1):
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
            
        result = _try_rank(r, n, m, p, T, 5, rng)
        if result:
            U_new, V_new, W_new, err = result
            if err < 1e-6:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = r
                break # Found a better one, stop searching lower? 
                # Or continue to find even lower?
                # Given time, break and return.
                
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
