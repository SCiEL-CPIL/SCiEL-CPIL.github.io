"""Rewritten program for matrix multiplication 4,5,6 rank minimization."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _compute_error(U, V, W):
    """Compute the max error using vectorized operations."""
    n, m, p = DIMENSIONS
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    
    U_sel = U_np[idx_u, :]
    V_sel = V_np[idx_v, :]
    W_sel = W_np[idx_w, :]
    
    vals = (U_sel * V_sel * W_sel).sum(axis=1)
    err = np.abs(vals - 1.0)
    return float(err.max()) if err.size > 0 else 0.0

def _compute_grad(U, V, W):
    """Compute gradients for U, V, W."""
    n, m, p = DIMENSIONS
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    
    U_sel = U_np[idx_u, :]
    V_sel = V_np[idx_v, :]
    W_sel = W_np[idx_w, :]
    
    vals = (U_sel * V_sel * W_sel).sum(axis=1)
    residuals = vals - 1.0
    
    VW = V_sel * W_sel
    UW = U_sel * W_sel
    UV = U_sel * V_sel
    
    dU = np.zeros_like(U_np)
    dV = np.zeros_like(V_np)
    dW = np.zeros_like(W_np)
    
    np.add.at(dU, idx_u, residuals[:, None] * VW)
    np.add.at(dV, idx_v, residuals[:, None] * UW)
    np.add.at(dW, idx_w, residuals[:, None] * UV)
    
    return dU, dV, dW, residuals

def _polish(U, V, W, max_iter=500, tol=1e-12, deadline=None):
    """High-precision polish using Newton's method on the tensor equations."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    N = n * m * p
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 5:
            break
        
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        
        vals = (U_sel * V_sel * W_sel).sum(axis=1)
        residuals = vals - 1.0
        
        err = np.abs(residuals).max()
        if err < tol:
            break
        
        # Build the Jacobian-like structure for a Gauss-Newton step
        # We solve: sum_c deltaU[idx_u,c]*V_sel*W_sel + ... = -residuals
        # This is a linear system. Use a simple iterative approach.
        
        # Compute the "Hessian" approximation
        VW = V_sel * W_sel  # (N, r)
        UW = U_sel * W_sel
        UV = U_sel * V_sel
        
        # For each factor, the update direction satisfies:
        # sum_c dU[idx_u,c] * VW[t,c] = -residuals[t]
        # This is an underdetermined system; use least squares via pseudo-inverse
        # but that's expensive. Instead, use a scaled gradient step with line search.
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        np.add.at(dU, idx_u, residuals[:, None] * VW)
        np.add.at(dV, idx_v, residuals[:, None] * UW)
        np.add.at(dW, idx_w, residuals[:, None] * UV)
        
        # Adaptive step size
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm < 1e-15:
            break
        
        # Try a few step sizes
        best_step = None
        best_err = err
        for factor in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
            step = factor * (err / (grad_norm + 1e-15))
            step = min(step, 1.0)
            
            U_new = U - step * dU
            V_new = V - step * dV
            W_new = W - step * dW
            
            U_sel_n = U_new[idx_u, :]
            V_sel_n = V_new[idx_v, :]
            W_sel_n = W_new[idx_w, :]
            vals_n = (U_sel_n * V_sel_n * W_sel_n).sum(axis=1)
            err_n = np.abs(vals_n - 1.0).max()
            
            if err_n < best_err:
                best_err = err_n
                best_step = step
        
        if best_step is None:
            break
        
        U -= best_step * dU
        V -= best_step * dV
        W -= best_step * dW
        
        if best_err < tol:
            break
    
    return U, V, W

def _try_rank_decomposition(target_rank, num_attempts=5, max_iter=200, deadline=None):
    """Try to find a rank decomposition using gradient descent with polish."""
    n, m, p = DIMENSIONS
    
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    N = n * m * p
    
    for attempt in range(num_attempts):
        if deadline and time.time() > deadline - 15:
            return None
        
        # Different initialization strategies
        if attempt % 3 == 0:
            seed = attempt * 7919 + target_rank
        elif attempt % 3 == 1:
            seed = attempt * 104729 + target_rank * 31
        else:
            seed = attempt * 15485863 + target_rank * 7
        
        np.random.seed(seed)
        
        # Initialize with small random values
        scale = 0.3
        U = np.random.randn(n*m, target_rank) * scale
        V = np.random.randn(m*p, target_rank) * scale
        W = np.random.random((p*n, target_rank)) * 2 - 1  # Uniform [-1, 1]
        
        # Scale to get initial values closer to 1
        # Target: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # With r terms, each product should be ~1/r
        U = U * (1.0 / (target_rank ** (1/3))) * 3
        V = V * (1.0 / (target_rank ** (1/3))) * 3
        
        lr = 0.1
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if deadline and time.time() > deadline - 15:
                break
            
            # Compute gradients
            dU, dV, dW, residuals = _compute_grad(U, V, W)
            err = np.abs(residuals).max()
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm > 1e-15:
                step = lr * (err / (grad_norm + 1e-15))
                step = min(step, 1.0)
            else:
                step = lr
            
            U -= step * dU
            V -= step * dV
            W -= step * dW
            
            # Decay learning rate
            if iteration % 30 == 0 and iteration > 0:
                lr *= 0.85
        
        # Polish the best solution found
        if best_err < 1e-4:
            U_p, V_p, W_p = _polish(best_U, best_V, best_W, max_iter=300, tol=1e-12, deadline=deadline)
            err_p = _compute_error(U_p, V_p, W_p)
            
            if err_p < best_err:
                best_err = err_p
                best_U, best_V, best_W = U_p, V_p, W_p
        
        # Try to round to integers/half-integers
        if best_err < 1e-6:
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            err_rounded = _compute_error(U_rounded, V_rounded, W_rounded)
            if err_rounded < 1e-10:
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
            
            # Also try integer rounding
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            err_int = _compute_error(U_int, V_int, W_int)
            if err_int < 1e-10:
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
    
    return None

def _strassen_like_init(n, m, p, rank):
    """Try to initialize using a Strassen-like block structure."""
    # This is a heuristic initialization that tries to exploit block structure
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Divide into 2x2 blocks if possible
    # For 4x5x6, we can think of it as blocks
    # This is just a heuristic to provide diverse initializations
    
    for c in range(rank):
        # Randomly assign some structure
        for i in range(n):
            for j in range(m):
                U[i*m+j, c] = np.random.choice([-1, 0, 1]) * np.random.uniform(0.5, 1.5)
        for j in range(m):
            for k in range(p):
                V[j*p+k, c] = np.random.choice([-1, 0, 1]) * np.random.uniform(0.5, 1.5)
        for k in range(p):
            for i in range(n):
                W[k*n+i, c] = np.random.choice([-1, 0, 1]) * np.random.uniform(0.5, 1.5)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s safety margin
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Focus on ranks that are known to be achievable or near-optimal
    # For 4x5x6, the minimum rank is at most 120 (schoolbook)
    # Try a range of ranks
    
    # First, try some promising ranks
    candidate_ranks = []
    for r in range(60, 120):
        candidate_ranks.append(r)
    
    # Also try some specific ranks that might work
    candidate_ranks = sorted(set(candidate_ranks))
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 30:
            break
        
        result = _try_rank_decomposition(target_rank, num_attempts=5, max_iter=200, deadline=deadline)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
