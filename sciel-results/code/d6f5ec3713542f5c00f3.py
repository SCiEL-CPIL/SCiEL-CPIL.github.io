import numpy as np
import time
import scipy.optimize as opt
import itertools

DIMENSIONS = [2, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for d in range(p):
                T[a*m+b, b*p+d, d*n+a] = 1.0
    
    # Compute decomposed tensor
    # Using einsum for efficiency
    T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    if np.max(np.abs(T_comp - T)) > tol:
        return False
    return True

def _make_schoolbook(n, m, p):
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _loss_and_grad(params, n, m, p, target_rank, T_target):
    """
    params: flattened vector of [U, V, W]
    """
    # Unpack
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    
    U = params[:size_U].reshape(n*m, target_rank)
    V = params[size_U:size_U+size_V].reshape(m*p, target_rank)
    W = params[size_U+size_V:].reshape(p*n, target_rank)
    
    # Compute T_comp
    T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
    diff = T_comp - T_target
    loss = np.sum(diff**2)
    
    # Gradients
    # dL/dU_i,r = 2 * sum_{j,k} diff_ijk * V_j,r * W_k,r
    # dL/dV_j,r = 2 * sum_{i,k} diff_ijk * U_i,r * W_k,r
    # dL/dW_k,r = 2 * sum_{i,j} diff_ijk * U_i,r * V_j,r
    
    # Efficient calculation using einsum
    # grad_U = 2 * einsum('ijk,jr,kr->ir', diff, V, W)
    # grad_V = 2 * einsum('ijk,ir,kr->jr', diff, U, W)
    # grad_W = 2 * einsum('ijk,ir,jr->kr', diff, U, V)
    
    grad_U = 2.0 * np.einsum('ijk,jr,kr->ir', diff, V, W)
    grad_V = 2.0 * np.einsum('ijk,ir,kr->jr', diff, U, W)
    grad_W = 2.0 * np.einsum('ijk,ir,jr->kr', diff, U, V)
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    return loss, grad

def _try_rank_reduction(n, m, p, target_rank, time_budget=30.0, n_restarts=5):
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < time_budget
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for d in range(p):
                T_target[a*m+b, b*p+d, d*n+a] = 1.0
    
    best_loss = float('inf')
    best_params = None
    
    for restart in range(n_restarts):
        if not still_going():
            break
        
        np.random.seed(42 + restart * 100)
        
        # Initialize with small random values
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use L-BFGS-B
        try:
            res = opt.minimize(
                _loss_and_grad, 
                params, 
                args=(n, m, p, target_rank, T_target),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
            )
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
                
                if best_loss < 1e-10:
                    break
                    
        except Exception:
            continue
            
    if best_loss < 1e-8 and best_params is not None:
        # Recover U, V, W
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        
        U = best_params[:size_U].reshape(n*m, target_rank)
        V = best_params[size_U:size_U+size_V].reshape(m*p, target_rank)
        W = best_params[size_U+size_V:].reshape(p*n, target_rank)
        
        # Try to snap to half-integers
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        if _verify(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
            return U_snapped, V_snapped, W_snapped
        
        # If snapping fails, check if the original is "close enough" to be considered valid 
        # but the task requires exact integer/half-integer. 
        # If it doesn't snap, we might return None to be safe, or try more iterations.
        # For now, return None if not snapped exactly.
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # safety margin
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Try to reduce rank
    # Known lower bounds suggest 45 is possible.
    # Try ranks from 45 up to best_rank - 1
    for rank in range(45, best_rank):
        if not still_going():
            break
        
        remaining = deadline - elapsed()
        if remaining < 15:
            break
        
        # Allocate more time for lower ranks
        budget = min(remaining * 0.8, 40.0)
        
        result = _try_rank_reduction(n, m, p, rank, time_budget=budget, n_restarts=10)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-9):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                # If we found a lower rank, we could break, but maybe an even lower one exists?
                # The loop goes 45, 46, ... so if 45 works, we return 45.
                # If 45 fails, we try 46.
                # We should break if we find the absolute best? No, we want the lowest.
                # Since we iterate from low to high, the first one found is the best.
                break
    
    return best
