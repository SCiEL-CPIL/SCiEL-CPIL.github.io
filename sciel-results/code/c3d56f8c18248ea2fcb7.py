import time
import numpy as np
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
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

def _snap_to_half_integers(mat, tol=1e-6):
    """Snap values to nearest half-integer if close."""
    snapped = np.round(mat * 2) / 2
    if np.max(np.abs(mat - snapped)) < tol:
        return snapped
    # Try integers
    snapped_int = np.round(mat)
    if np.max(np.abs(mat - snapped_int)) < tol:
        return snapped_int
    return mat

def _polish(U, V, W, T_target, max_iter=100, tol=1e-12):
    """Use scipy.optimize to polish the solution to high precision."""
    dim_U, rank = U.shape
    dim_V, _ = V.shape
    dim_W, _ = W.shape
    
    # Flatten U, V, W into a single vector
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(x):
        U_flat = x[:dim_U*rank].reshape(dim_U, rank)
        V_flat = x[dim_U*rank:dim_U*rank+dim_V*rank].reshape(dim_V, rank)
        W_flat = x[dim_U*rank+dim_V*rank:].reshape(dim_W, rank)
        
        T_recon = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
        # Return the flattened residual
        return (T_recon - T_target).flatten()
    
    try:
        result = scipy.optimize.least_squares(objective, x0, method='lm', max_nfev=max_iter)
        if result.success or np.linalg.norm(result.fun) < tol:
            U_new = result.x[:dim_U*rank].reshape(dim_U, rank)
            V_new = result.x[dim_U*rank:dim_U*rank+dim_V*rank].reshape(dim_V, rank)
            W_new = result.x[dim_U*rank+dim_V*rank:].reshape(dim_W, rank)
            return U_new, V_new, W_new
    except Exception:
        pass
    
    return U, V, W

def _try_als(rank, n, m, p, max_iter=100, seed=42, time_limit=5):
    """Try alternating least squares to find rank decomposition."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    rng = np.random.RandomState(seed)
    
    # Initialize
    # Use a smaller scale to avoid overflow in early iterations
    U = rng.randn(dim_U, rank) * 0.1
    V = rng.randn(dim_V, rank) * 0.1
    W = rng.randn(dim_W, rank) * 0.1
    
    # Target tensor
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() > start_time + time_limit:
            break
        
        # Update U
        S_reshaped = np.zeros((rank, dim_V * dim_W))
        for r in range(rank):
            S_reshaped[r, :] = np.outer(V[:, r], W[:, r]).flatten()
        try:
            # Solve T_target.reshape(dim_U, -1) = U @ S_reshaped
            # U = T_target.reshape(dim_U, -1) @ S_reshaped^+
            U_new, _, _, _ = np.linalg.lstsq(S_reshaped.T, T_target.reshape(dim_U, -1).T, rcond=None)
            U_new = U_new.T
        except:
            break
        
        # Update V
        S_UW_reshaped = np.zeros((rank, dim_U * dim_W))
        for r in range(rank):
            S_UW_reshaped[r, :] = np.outer(U[:, r], W[:, r]).flatten()
        try:
            # T_target.transpose(1, 0, 2).reshape(dim_V, dim_U*dim_W) = V @ S_UW_reshaped
            V_new, _, _, _ = np.linalg.lstsq(S_UW_reshaped.T, T_target.transpose(1, 0, 2).reshape(dim_V, dim_U * dim_W).T, rcond=None)
            V_new = V_new.T
        except:
            break
        
        # Update W
        S_UV_reshaped = np.zeros((rank, dim_U * dim_V))
        for r in range(rank):
            S_UV_reshaped[r, :] = np.outer(U[:, r], V[:, r]).flatten()
        try:
            # T_target.transpose(2, 0, 1).reshape(dim_W, dim_U*dim_V) = W @ S_UV_reshaped
            W_new, _, _, _ = np.linalg.lstsq(S_UV_reshaped.T, T_target.transpose(2, 0, 1).reshape(dim_W, dim_U * dim_V).T, rcond=None)
            W_new = W_new.T
        except:
            break
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T_target))
        if err < 1e-6:
            # If we get close, try to polish
            U, V, W = _polish(U, V, W, T_target, max_iter=50)
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T_target))
            if err < 1e-9:
                break
    
    # Try snapping
    U_snap = _snap_to_half_integers(U)
    V_snap = _snap_to_half_integers(V)
    W_snap = _snap_to_half_integers(W)
    
    if _verify(U_snap, V_snap, W_snap, n, m, p):
        return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
    
    # Also try without snapping (in case it's exact float)
    if _verify(U, V, W, n, m, p):
        return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def construct():
    n, m, p = 2, 3, 4
    
    # Try ranks from 13 to 23
    for rank in range(13, 24):
        if time.time() > _DEADLINE - 10:
            break
        
        # Try ALS with multiple seeds
        # Rank 13 is the target, give it more seeds
        max_seeds = 50 if rank == 13 else 20
        for seed in range(rank * 100, rank * 100 + max_seeds):
            if time.time() > _DEADLINE - 5:
                break
            res = _try_als(rank, n, m, p, max_iter=100, seed=seed, time_limit=2)
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
