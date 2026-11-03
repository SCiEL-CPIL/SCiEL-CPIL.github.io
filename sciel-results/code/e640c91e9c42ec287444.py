import time
import numpy as np
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 130  # 20s safety margin

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if U,V,W form a valid decomposition."""
    if not U or not V or not W:
        return False
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
        
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
        
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

def _get_schoolbook(n, m, p):
    """Get the standard schoolbook decomposition."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    return U, V, W

def _residual(x, n, m, p, R):
    """
    Compute residual for least_squares.
    x: flattened vector of U, V, W
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Unpack
    U = x[:dim_U * R].reshape(dim_U, R)
    V = x[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
    W = x[dim_U * R + dim_V * R :].reshape(dim_W, R)
    
    # Reconstruct
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Target
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    return (T_recon - T_target).flatten()

def _try_rank_ls(rank, n, m, p, max_seeds=10, seed_base=42, time_limit=10):
    """Try to find a rank-rank decomposition via Least Squares."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    total_vars = (dim_U + dim_V + dim_W) * rank
    
    # Pre-compute schoolbook for initialization
    U_sb, V_sb, W_sb = _get_schoolbook(n, m, p)
    
    for seed in range(seed_base, seed_base + max_seeds):
        if time.time() > _DEADLINE:
            return None
            
        rng = np.random.RandomState(seed)
        
        # Initialization Strategy:
        # 1. Use the first 'rank' columns of the schoolbook decomposition.
        # This is a strong warm start.
        # We take the first 'rank' terms.
        # If rank < 24, we are dropping some terms. The optimizer must adjust the remaining terms to cover the full tensor.
        
        U_init = np.zeros((dim_U, rank))
        V_init = np.zeros((dim_V, rank))
        W_init = np.zeros((dim_W, rank))
        
        # Copy first 'rank' schoolbook terms
        for r in range(rank):
            U_init[:, r] = U_sb[:, r]
            V_init[:, r] = V_sb[:, r]
            W_init[:, r] = W_sb[:, r]
            
        # Add small noise to avoid local minima if the warm start is too perfect but stuck?
        # Actually, if it's a valid decomposition, no noise is needed.
        # But for lower ranks, the warm start is NOT a valid decomposition.
        # The noise helps escape symmetries.
        
        noise_level = 0.1
        U_init += rng.randn(dim_U, rank) * noise_level
        V_init += rng.randn(dim_V, rank) * noise_level
        W_init += rng.randn(dim_W, rank) * noise_level
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = scipy.optimize.least_squares(
                _residual, x0, args=(n, m, p, rank),
                method='lm', # Levenberg-Marquardt is fast for small problems
                ftol=1e-12, xtol=1e-12, gtol=1e-12
            )
        except Exception:
            continue
            
        if res.success or res.cost < 1e-10:
            x_sol = res.x
            U = x_sol[:dim_U * rank].reshape(dim_U, rank)
            V = x_sol[dim_U * rank : dim_U * rank + dim_V * rank].reshape(dim_V, rank)
            W = x_sol[dim_U * rank + dim_V * rank :].reshape(dim_W, rank)
            
            # Try to snap to half-integers
            # Strategy 1: Round to nearest half-integer
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if _verify(U_snap, V_snap, W_snap, n, m, p):
                return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
                
            # Strategy 2: Round to nearest integer
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if _verify(U_int, V_int, W_int, n, m, p):
                return (U_int.tolist(), V_int.tolist(), W_int.tolist())
                
    return None

def construct():
    n, m, p = 2, 3, 4
    
    # Prioritize ranks 22 and 23, as they are more likely to exist and be found
    # Rank 20 and 21 are likely infeasible or much harder.
    for rank in [22, 23]:
        if time.time() > _DEADLINE - 10:
            break
            
        # More seeds for these ranks
        seeds = 100 if rank == 22 else 50
        
        res = _try_rank_ls(rank, n, m, p, max_seeds=seeds, seed_base=rank * 1000)
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
