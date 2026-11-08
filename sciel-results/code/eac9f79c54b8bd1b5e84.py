import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _construct_rank_55():
    """
    Hardcoded rank-55 decomposition for 2x3x10 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k  # Wait, V index is m*p. B is m x p. Index is j*p + k? 
                # In the problem, B is 3x10. m=3, p=10.
                # V shape is (m*p, rank). Index is j*p + k?
                # Let's check the seed code.
                # V[0 * 10 + k0]. Yes, row index is j * p + k.
                # W shape is (p*n, rank). Index is k * n + i.
                # W[k0 * 2 + 0]. Yes, row index is k * n + i.
                # U shape is (n*m, rank). Index is i * m + j.
                # U[0 * 3 + 0]. Yes, row index is i * m + j.
                
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-7):
    """Check if U, V, W exactly decompose T."""
    # Reconstruct tensor
    # T_recon[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
    # This is a tensor contraction.
    # We can do this by iterating over r or using einsum.
    # np.einsum('ur,vr,wr->uvw', U, V, W)
    T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.allclose(T_recon, T, atol=tol, rtol=tol)

def _try_find_rank(T, rank, n, m, p, time_budget):
    """
    Attempt to find a decomposition of rank `rank` for tensor T.
    Returns (U, V, W) if found, else None.
    """
    start = time.time()
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Number of variables
    num_vars = rank * (dim_u + dim_v + dim_w)
    
    # Objective function
    def objective(params):
        U = params[:rank * dim_u].reshape(dim_u, rank)
        V = params[rank * dim_u : rank * dim_u + rank * dim_v].reshape(dim_v, rank)
        W = params[rank * dim_u + rank * dim_v :].reshape(dim_w, rank)
        
        # Compute error
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        error = T_recon - T
        return np.sum(error ** 2)
    
    # Gradient (optional, but L-BFGS-B can estimate it)
    # For speed, we might use a derivative-free method if gradient is too slow, 
    # but L-BFGS-B is usually fast enough for this size.
    
    best_result = None
    best_score = float('inf')
    
    # Multiple random restarts
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() - start > time_budget:
            break
        
        # Initialize parameters
        # Scale matters. Let's start with small random values.
        init_params = np.random.randn(num_vars) * 0.1
        
        try:
            # Use L-BFGS-B
            res = opt.minimize(objective, init_params, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                
                # Check if we found a valid solution
                if best_score < 1e-8:
                    U = best_result[:rank * dim_u].reshape(dim_u, rank)
                    V = best_result[rank * dim_u : rank * dim_u + rank * dim_v].reshape(dim_v, rank)
                    W = best_result[rank * dim_u + rank * dim_v :].reshape(dim_w, rank)
                    
                    # Round to nearest integer or half-integer?
                    # The problem allows integer/half-integer.
                    # Let's try rounding to nearest integer first.
                    U_int = np.rint(U).astype(int)
                    V_int = np.rint(V).astype(int)
                    W_int = np.rint(W).astype(int)
                    
                    if _verify_decomposition(U_int, V_int, W_int, T, tol=1e-7):
                        return U_int, V_int, W_int
                    
                    # Try rounding to nearest half-integer (multiply by 2, round, divide by 2)
                    U_half = np.rint(U * 2) / 2
                    V_half = np.rint(V * 2) / 2
                    W_half = np.rint(W * 2) / 2
                    
                    if _verify_decomposition(U_half, V_half, W_half, T, tol=1e-7):
                        return U_half, V_half, W_half
                        
        except Exception as e:
            pass
            
    return None

def _try_drop_column(U, V, W, T, col_idx, time_budget):
    """
    Try to drop column col_idx from the decomposition and re-optimize.
    Returns (U_new, V_new, W_new) if successful, else None.
    """
    start = time.time()
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    current_rank = U.shape[1]
    new_rank = current_rank - 1
    
    if new_rank < 1:
        return None
    
    # Remove the column
    mask = np.ones(current_rank, dtype=bool)
    mask[col_idx] = False
    U_init = U[:, mask]
    V_init = V[:, mask]
    W_init = W[:, mask]
    
    # Flatten for optimization
    params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def objective(params):
        U = params[:new_rank * dim_u].reshape(dim_u, new_rank)
        V = params[new_rank * dim_u : new_rank * dim_u + new_rank * dim_v].reshape(dim_v, new_rank)
        W = params[new_rank * dim_u + new_rank * dim_v :].reshape(dim_w, new_rank)
        
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        error = T_recon - T
        return np.sum(error ** 2)
    
    try:
        res = opt.minimize(objective, params_init, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.fun < 1e-8:
            U_new = res.x[:new_rank * dim_u].reshape(dim_u, new_rank)
            V_new = res.x[new_rank * dim_u : new_rank * dim_u + new_rank * dim_v].reshape(dim_v, new_rank)
            W_new = res.x[new_rank * dim_u + new_rank * dim_v :].reshape(dim_w, new_rank)
            
            # Try rounding
            U_int = np.rint(U_new).astype(int)
            V_int = np.rint(V_new).astype(int)
            W_int = np.rint(W_new).astype(int)
            
            if _verify_decomposition(U_int, V_int, W_int, T, tol=1e-7):
                return U_int, V_int, W_int
            
            U_half = np.rint(U_new * 2) / 2
            V_half = np.rint(V_new * 2) / 2
            W_half = np.rint(W_new * 2) / 2
            
            if _verify_decomposition(U_half, V_half, W_half, T, tol=1e-7):
                return U_half, V_half, W_half
                
    except Exception:
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin
    
    # Step 1: Get baseline rank 55
    U55, V55, W55 = _construct_rank_55()
    best_rank = 55
    best_U, best_V, best_W = U55, V55, W55
    
    # Step 2: Build target tensor
    T = _build_target_tensor()
    
    # Step 3: Try column dropping from rank 55 to get rank 54, 53, etc.
    # This is more targeted than random restarts
    current_U, current_V, current_W = U55.copy(), V55.copy(), W55.copy()
    current_rank = 55
    
    # Try dropping columns one by one
    while current_rank > 50:
        if time.time() > DEADLINE - 15:
            break
        
        dropped = False
        # Try each column, but limit the number of attempts per rank level
        max_col_attempts = 10
        cols_to_try = list(range(current_rank))
        np.random.shuffle(cols_to_try)
        
        for col_idx in cols_to_try[:max_col_attempts]:
            if time.time() > DEADLINE - 15:
                break
            
            time_per_drop = min(5.0, (DEADLINE - time.time()) / (current_rank + 1))
            result = _try_drop_column(current_U, current_V, current_W, T, col_idx, time_per_drop)
            
            if result is not None:
                current_U, current_V, current_W = result
                current_rank -= 1
                best_rank = current_rank
                best_U, best_V, best_W = current_U, current_V, current_W
                dropped = True
                break
        
        if not dropped:
            # No more columns can be dropped, break out
            break
    
    # If we still have time, try random restarts for even lower ranks
    if time.time() < DEADLINE - 20:
        for r in range(current_rank - 1, max(current_rank - 5, 49), -1):
            if time.time() > DEADLINE - 10:
                break
            
            time_per_attempt = min(8.0, (DEADLINE - time.time()) / 3)
            result = _try_find_rank(T, r, n, m, p, time_per_attempt)
            
            if result is not None:
                best_rank = r
                best_U, best_V, best_W = result
                break
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
