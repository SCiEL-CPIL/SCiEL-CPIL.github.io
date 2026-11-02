import time
import numpy as np
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if U,V,W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Construct target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    # Compute reconstructed tensor
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check difference
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _objective(params, T_target, n, m, p, rank):
    """
    Objective function for least_squares.
    params: flattened [U, V, W]
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    size_U = dim_U * rank
    size_V = dim_V * rank
    
    U_flat = params[:size_U]
    V_flat = params[size_U:size_U + size_V]
    W_flat = params[size_U + size_V:]
    
    U = U_flat.reshape(dim_U, rank)
    V = V_flat.reshape(dim_V, rank)
    W = W_flat.reshape(dim_W, rank)
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    # Return the residual vector
    return (T_recon - T_target).flatten()

def _try_rank_opt(rank, n, m, p, max_time=10, seed=42):
    """Try to find a rank-rank decomposition via least_squares."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    total_vars = (dim_U + dim_V + dim_W) * rank
    
    rng = np.random.RandomState(seed)
    
    # Try a few restarts
    for restart in range(5):
        if time.time() > _DEADLINE - 2:
            break
            
        # Initialize with small random values
        x0 = rng.randn(total_vars) * 0.5
        
        try:
            # Use least_squares
            result = scipy.optimize.least_squares(
                _objective, 
                x0, 
                args=(T_target, n, m, p, rank),
                method='lm', # Levenberg-Marquardt is good for small problems
                max_nfev=2000
            )
            
            if result.success or result.cost < 1e-8:
                # Extract U, V, W
                size_U = dim_U * rank
                size_V = dim_V * rank
                
                U_flat = result.x[:size_U]
                V_flat = result.x[size_U:size_U + size_V]
                W_flat = result.x[size_U + size_V:]
                
                U = U_flat.reshape(dim_U, rank)
                V = V_flat.reshape(dim_V, rank)
                W = W_flat.reshape(dim_W, rank)
                
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                    return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
                    
                # Try integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
                    return (U_int.tolist(), V_int.tolist(), W_int.tolist())
                    
        except Exception as e:
            pass
            
    return None

def construct():
    n, m, p = 2, 3, 4
    
    # Try ranks from 18 to 24
    # 18 is the known minimal rank for 2x3x4
    for rank in range(18, 25):
        if time.time() > _DEADLINE - 5:
            break
            
        # Try multiple seeds
        for seed in range(20):
            if time.time() > _DEADLINE - 2:
                break
            res = _try_rank_opt(rank, n, m, p, seed=seed)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
                
    # Fallback: Schoolbook
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
