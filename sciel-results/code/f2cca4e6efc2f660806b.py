import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

def _make_schoolbook():
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

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_half_integer(U, V, W):
    # Check if all entries are close to half-integers
    for M in [U, V, W]:
        for row in M:
            for val in row:
                # val should be close to k * 0.5
                if abs(val * 2 - round(val * 2)) > 1e-6:
                    return False
    return True

def _round_to_half_integer(M):
    M_rounded = [[round(val * 2) / 2.0 for val in row] for row in M]
    return M_rounded

def _search_decomposition(target_rank, time_limit=100):
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We want to minimize sum (T - UVW)^2
    # T is sparse.
    
    def objective_and_grad(params):
        # params is a flat array of size (size_u + size_v + size_w) * target_rank
        U_flat = params[:size_u * target_rank]
        V_flat = params[size_u * target_rank : size_u * target_rank + size_v * target_rank]
        W_flat = params[size_u * target_rank + size_v * target_rank :]
        
        U = U_flat.reshape(size_u, target_rank)
        V = V_flat.reshape(size_v, target_rank)
        W = W_flat.reshape(size_w, target_rank)
        
        # Compute error
        # We only need to check the non-zero entries of T
        # T[i*m+j, j*p+k, k*n+i] = 1
        error = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Precompute sums for efficiency?
        # For each c, we have u_c, v_c, w_c.
        # The term is sum_c u_{a,c} v_{b,c} w_{d,c}.
        
        # Iterate over all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    d = k * n + i
                    
                    # Compute S = sum_c U[a,c] V[b,c] W[d,c]
                    S = 0.0
                    # Gradient contribution for this (a,b,d)
                    # dE/dU[a,c] += 2 (S - 1) V[b,c] W[d,c]
                    # dE/dV[b,c] += 2 (S - 1) U[a,c] W[d,c]
                    # dE/dW[d,c] += 2 (S - 1) U[a,c] V[b,c]
                    
                    for c in range(target_rank):
                        term = U[a, c] * V[b, c] * W[d, c]
                        S += term
                        
                    diff = S - 1.0
                    factor = 2.0 * diff
                    
                    for c in range(target_rank):
                        grad_U[a, c] += factor * V[b, c] * W[d, c]
                        grad_V[b, c] += factor * U[a, c] * W[d, c]
                        grad_W[d, c] += factor * U[a, c] * V[b, c]
                        
                    error += diff * diff
        
        # Also, we need to ensure that other entries are 0?
        # The tensor T is 0 elsewhere.
        # If we don't constrain them, they might be non-zero.
        # However, if we find a decomposition of the support, it might not be exact for the full tensor.
        # But usually, if the support is satisfied, and the rank is low, it might work?
        # No, we MUST satisfy the zero entries.
        # Adding all zero entries to the objective is expensive (900-30 = 870 entries).
        # But it's necessary for correctness.
        
        # Let's add a penalty for non-zero entries.
        # Or just compute the full tensor product and subtract T.
        # Full tensor size: 12 * 20 * 15 = 3600 entries.
        # This is small enough to compute fully.
        
        # Let's re-calculate using full tensor to be safe.
        # T_full is sparse.
        
        # Actually, the gradient above only handles the 1s.
        # We need gradients for the 0s too.
        # T_{a,b,d} = 0 for (a,b,d) not in the support.
        
        # Let's stick to the support only for now and hope the null space is handled?
        # No, that's dangerous.
        
        # Let's include the zeros.
        # It's just 870 more iterations.
        # Total 900 iterations.
        # Inner loop 18.
        # 900 * 18 = 16200 ops per gradient step.
        # This is fast.
        
        # I will rewrite the loop to cover all a,b,d.
        
        # But I already wrote the loop for 1s.
        # I'll add a second loop for 0s.
        
        # Identify support
        support = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    support.add((i*m+j, j*p+k, k*n+i))
                    
        all_indices = range(size_u * size_v * size_w)
        # This is too slow to generate all and check set.
        
        # Alternative:
        # Compute the full reconstructed tensor R
        # R = np.einsum('ac,bc,dc->abd', U, V, W)
        # T is sparse.
        # Error = ||R - T||^2
        
        # Let's use numpy for the objective and finite difference for gradient?
        # No, analytic gradient is better.
        
        # I will use the full tensor approach in the objective.
        
        R = np.einsum('ac,bc,dc->abd', U, V, W)
        
        # Construct T
        T = np.zeros((size_u, size_v, size_w))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        diff = R - T
        error = np.sum(diff**2)
        
        # Gradient
        # dE/dU[a,c] = 2 sum_{b,d} diff[a,b,d] V[b,c] W[d,c]
        # dE/dV[b,c] = 2 sum_{a,d} diff[a,b,d] U[a,c] W[d,c]
        # dE/dW[d,c] = 2 sum_{a,b} diff[a,b,d] U[a,c] V[b,c]
        
        grad_U = 2 * np.einsum('abd,bc,dc->ac', diff, V, W)
        grad_V = 2 * np.einsum('abd,ac,dc->bc', diff, U, W)
        grad_W = 2 * np.einsum('abd,ac,bc->dc', diff, U, V)
        
        grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return error, grad_flat

    start_time = time.time()
    best_err = float('inf')
    best_params = None
    
    while time.time() - start_time < time_limit:
        # Random init
        scale = 1.0
        U_init = np.random.randn(size_u, target_rank) * scale
        V_init = np.random.randn(size_v, target_rank) * scale
        W_init = np.random.randn(size_w, target_rank) * scale
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Optimize
        try:
            from scipy.optimize import minimize
            res = minimize(objective_and_grad, params_init, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-10})
        except Exception:
            break
            
        if res.success or res.fun < best_err:
            best_err = res.fun
            best_params = res.x
            
        if best_err < 1e-6:
            break
            
    if best_params is None:
        return None
        
    U_opt = best_params[:size_u * target_rank].reshape(size_u, target_rank)
    V_opt = best_params[size_u * target_rank : size_u * target_rank + size_v * target_rank].reshape(size_v, target_rank)
    W_opt = best_params[size_u * target_rank + size_v * target_rank :].reshape(size_w, target_rank)
    
    # Round to half-integers
    U_rounded = _round_to_half_integer(U_opt.tolist())
    V_rounded = _round_to_half_integer(V_opt.tolist())
    W_rounded = _round_to_half_integer(W_opt.tolist())
    
    # Verify
    if _verify(U_rounded, V_rounded, W_rounded):
        if _verify_half_integer(U_rounded, V_rounded, W_rounded):
            return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Try rank 36
    # This corresponds to 2 * 18
    result = _search_decomposition(36, time_limit=110)
    
    if result is not None:
        return result
        
    # Fallback to schoolbook
    return _make_schoolbook()
