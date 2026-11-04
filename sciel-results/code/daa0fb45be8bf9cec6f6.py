"""Find a rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]
SEED_RANK = 72  # 3*4*6

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _to_half_integer_check(mat, tol=1e-6):
    """Check if all entries are integer or half-integer within tolerance."""
    scaled = 2 * np.asarray(mat, dtype=float)
    rounded = np.round(scaled)
    return np.max(np.abs(scaled - rounded)) < tol

def _validate(U, V, W, n, m, p, rank, tol=1e-6):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # Check half-integer constraint
    if not _to_half_integer_check(U) or not _to_half_integer_check(V) or not _to_half_integer_check(W):
        return False
    
    # Verify tensor decomposition:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _validate_fast(U, V, W, n, m, p, rank, tol=1e-6):
    """Fast vectorized validation using matrix operations."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is equivalent to: for each c, we have outer products
    # Let's compute it as: for each c, U_c (n*m) x V_c (m*p) x W_c (p*n)
    # Actually, let's think of it differently.
    
    # The condition is: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # 
    # Let's define for each c:
    #   U_c has shape (n, m) with U_c[i,j] = U[i*m+j, c]
    #   V_c has shape (m, p) with V_c[j,k] = V[j*p+k, c]  
    #   W_c has shape (p, n) with W_c[k,i] = W[k*n+i, c]
    #
    # Then the condition is: sum_c sum over... wait, it's element-wise product summed over c.
    # For fixed (i,j,k): sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
    #
    # We can compute this by: for each c, compute the tensor U_c ⊙ V_c ⊙ W_c (element-wise in the appropriate sense)
    # Actually, let's just do it with einsum or explicit loops for small dimensions.
    
    # Reshape
    U_reshaped = U.reshape(n, m, rank)  # U_reshaped[i,j,c]
    V_reshaped = V.reshape(m, p, rank)  # V_reshaped[j,k,c]
    W_reshaped = W.reshape(p, n, rank)  # W_reshaped[k,i,c]
    
    # Compute sum over c of U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    # This is: np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    try:
        result = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        # Fallback to loops
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i,j,k] = np.sum(U_reshaped[i,j,:] * V_reshaped[j,k,:] * W_reshaped[k,i,:])
    
    return np.all(np.abs(result - 1.0) < tol)

def _objective(params, n, m, p, rank):
    """Objective function for optimization: minimize sum of squared errors."""
    U = params[0:rank*n*m].reshape(n*m, rank)
    V = params[rank*n*m:rank*n*m + rank*m*p].reshape(m*p, rank)
    W = params[rank*n*m + rank*m*p:].reshape(p*n, rank)
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    try:
        result = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i,j,k] = np.sum(U_reshaped[i,j,:] * V_reshaped[j,k,:] * W_reshaped[k,i,:])
    
    error = result - 1.0
    return np.sum(error * error)

def _polish_with_rounding(U, V, W, n, m, p, rank, tol=1e-6):
    """Try to polish by rounding to half-integers and check if it still works."""
    U_rounded = np.round(2 * U) / 2
    V_rounded = np.round(2 * V) / 2
    W_rounded = np.round(2 * W) / 2
    
    if _validate_fast(U_rounded, V_rounded, W_rounded, n, m, p, rank, tol=1e-4):
        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    return None

def _try_rank(rank, n, m, p, time_budget, seed=42):
    """Try to find a decomposition of given rank using optimization."""
    np.random.seed(seed)
    
    # Initialize with schoolbook-like structure but with fewer terms
    # Start with random half-integer values
    U = np.random.randint(0, 3, (n*m, rank)) / 2.0
    V = np.random.randint(0, 3, (m*p, rank)) / 2.0
    W = np.random.randint(0, 3, (p*n, rank)) / 2.0
    
    # Check if random init works (unlikely but check)
    if _validate_fast(U, V, W, n, m, p, rank, tol=1e-4):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Use gradient-based optimization
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    start_time = time.time()
    max_iter = 5000
    tol = 1e-8
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Compute objective
        obj_val = _objective(params, n, m, p, rank)
        
        if obj_val < tol:
            # Check if we can round to half-integers
            U_opt = params[0:rank*n*m].reshape(n*m, rank)
            V_opt = params[rank*n*m:rank*n*m + rank*m*p].reshape(m*p, rank)
            W_opt = params[rank*n*m + rank*m*p:].reshape(p*n, rank)
            
            polished = _polish_with_rounding(U_opt, V_opt, W_opt, n, m, p, rank)
            if polished:
                return polished
            
            # Also check without rounding
            if _validate_fast(U_opt, V_opt, W_opt, n, m, p, rank, tol=1e-4):
                return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
            break
        
        # Compute gradient numerically (for small problems this is feasible)
        # Use finite differences with step size
        step = 1e-5
        grad = np.zeros_like(params)
        
        # Only compute gradient for a subset of parameters to save time
        # For larger problems, use automatic differentiation or sparse gradients
        num_params = len(params)
        
        if num_params <= 5000:
            for idx in range(num_params):
                orig = params[idx]
                params[idx] = orig + step
                obj_plus = _objective(params, n, m, p, rank)
                params[idx] = orig - step
                obj_minus = _objective(params, n, m, p, rank)
                params[idx] = orig
                grad[idx] = (obj_plus - obj_minus) / (2 * step)
        else:
            # For larger problems, use a simpler gradient approximation
            # Just use the current direction and take a step
            U_cur = params[0:rank*n*m].reshape(n*m, rank)
            V_cur = params[rank*n*m:rank*n*m + rank*m*p].reshape(m*p, rank)
            W_cur = params[rank*n*m + rank*m*p:].reshape(p*n, rank)
            
            U_reshaped = U_cur.reshape(n, m, rank)
            V_reshaped = V_cur.reshape(m, p, rank)
            W_reshaped = W_cur.reshape(p, n, rank)
            
            try:
                result = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            except Exception:
                result = np.zeros((n, m, p))
            
            error = result - 1.0
            
            # Approximate gradients
            grad_U = np.zeros_like(U_cur)
            grad_V = np.zeros_like(V_cur)
            grad_W = np.zeros_like(W_cur)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            # d/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c] * error[i,j,k]
                            grad_U[i*m+j, c] += V_cur[j*p+k, c] * W_cur[k*n+i, c] * error[i,j,k]
                            # d/dV[j*p+k,c] = U[i*m+j,c] * W[k*n+i,c] * error[i,j,k]
                            grad_V[j*p+k, c] += U_cur[i*m+j, c] * W_cur[k*n+i, c] * error[i,j,k]
                            # d/dW[k*n+i,c] = U[i*m+j,c] * V[j*p+k,c] * error[i,j,k]
                            grad_W[k*n+i, c] += U_cur[i*m+j, c] * V_cur[j*p+k, c] * error[i,j,k]
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        # Adaptive learning rate
        lr = 0.01 / (1 + iteration * 0.001)
        params -= lr * grad
    
    # Final check
    U_opt = params[0:rank*n*m].reshape(n*m, rank)
    V_opt = params[rank*n*m:rank*n*m + rank*m*p].reshape(m*p, rank)
    W_opt = params[rank*n*m + rank*m*p:].reshape(p*n, rank)
    
    polished = _polish_with_rounding(U_opt, V_opt, W_opt, n, m, p, rank)
    if polished:
        return polished
    
    if _validate_fast(U_opt, V_opt, W_opt, n, m, p, rank, tol=1e-4):
        return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe within 150s
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition as fallback
    fallback = _schoolbook()
    
    best_result = fallback
    best_rank = SEED_RANK
    
    # Try ranks from 71 down to 65
    for target_rank in range(71, 64, -1):
        if time.time() > deadline:
            break
        
        # Try a few different seeds
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline:
                break
            
            remaining_time = deadline - time.time()
            if remaining_time < 10:
                break
            
            result = _try_rank(target_rank, n, m, p, time_budget=remaining_time * 0.8, seed=seed)
            if result is not None:
                # Validate the result
                U_arr = np.array(result['U'])
                V_arr = np.array(result['V'])
                W_arr = np.array(result['W'])
                
                if _validate(U_arr, V_arr, W_arr, n, m, p, target_rank, tol=1e-4):
                    best_result = result
                    best_rank = target_rank
                    break
        
        if best_rank < SEED_RANK:
            break
    
    # Check time
    if time.time() > deadline:
        return best_result
    
    return best_result
