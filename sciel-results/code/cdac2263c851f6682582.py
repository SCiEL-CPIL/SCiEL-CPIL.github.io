import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 135  # 15s safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-8):
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

def _snap_to_half_integers(mat, tol=1e-5):
    """Snap values to nearest half-integer if close."""
    snapped = np.round(mat * 2) / 2
    if np.max(np.abs(mat - snapped)) < tol:
        return snapped
    return mat

def _rational_reconstruct(val, max_denom=100):
    """Try to find a rational number p/q close to val with small denominator."""
    # First try half-integers
    half = np.round(val * 2) / 2
    if abs(val - half) < 1e-6:
        return half
    
    # Try integers
    int_val = np.round(val)
    if abs(val - int_val) < 1e-6:
        return int_val
    
    # Try simple rationals
    for q in range(1, max_denom + 1):
        p = round(val * q)
        if abs(val - p / q) < 1e-6:
            return p / q
    return val

def _polish(U, V, W, T_target, max_iter=200, tol=1e-12):
    """Use iterative refinement to polish the solution to high precision."""
    dim_U, rank = U.shape
    dim_V, _ = V.shape
    dim_W, _ = W.shape
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for _ in range(max_iter):
        # Update U
        S_reshaped = np.zeros((rank, dim_V * dim_W))
        for r in range(rank):
            S_reshaped[r, :] = np.outer(V[:, r], W[:, r]).flatten()
        try:
            U_new, _, _, _ = np.linalg.lstsq(S_reshaped.T, T_target.reshape(dim_U, -1).T, rcond=None)
            U_new = U_new.T
        except:
            break
        
        # Update V
        S_UW_reshaped = np.zeros((rank, dim_U * dim_W))
        for r in range(rank):
            S_UW_reshaped[r, :] = np.outer(U_new[:, r], W[:, r]).flatten()
        try:
            V_new, _, _, _ = np.linalg.lstsq(S_UW_reshaped.T, T_target.transpose(1, 0, 2).reshape(dim_V, dim_U * dim_W).T, rcond=None)
            V_new = V_new.T
        except:
            break
        
        # Update W
        S_UV_reshaped = np.zeros((rank, dim_U * dim_V))
        for r in range(rank):
            S_UV_reshaped[r, :] = np.outer(U_new[:, r], V_new[:, r]).flatten()
        try:
            W_new, _, _, _ = np.linalg.lstsq(S_UV_reshaped.T, T_target.transpose(2, 0, 1).reshape(dim_W, dim_U * dim_V).T, rcond=None)
            W_new = W_new.T
        except:
            break
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T_target))
        if err < tol:
            break
    
    return U, V, W

def _try_als(rank, n, m, p, max_iter=200, seed=42, time_limit=3):
    """Try alternating least squares to find rank decomposition."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    rng = np.random.RandomState(seed)
    
    # Initialize
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
            U_new, _, _, _ = np.linalg.lstsq(S_reshaped.T, T_target.reshape(dim_U, -1).T, rcond=None)
            U_new = U_new.T
        except:
            break
        
        # Update V
        S_UW_reshaped = np.zeros((rank, dim_U * dim_W))
        for r in range(rank):
            S_UW_reshaped[r, :] = np.outer(U_new[:, r], W[:, r]).flatten()
        try:
            V_new, _, _, _ = np.linalg.lstsq(S_UW_reshaped.T, T_target.transpose(1, 0, 2).reshape(dim_V, dim_U * dim_W).T, rcond=None)
            V_new = V_new.T
        except:
            break
        
        # Update W
        S_UV_reshaped = np.zeros((rank, dim_U * dim_V))
        for r in range(rank):
            S_UV_reshaped[r, :] = np.outer(U_new[:, r], V_new[:, r]).flatten()
        try:
            W_new, _, _, _ = np.linalg.lstsq(S_UV_reshaped.T, T_target.transpose(2, 0, 1).reshape(dim_W, dim_U * dim_V).T, rcond=None)
            W_new = W_new.T
        except:
            break
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T_target))
        if err < 1e-8:
            # If we get close, try to polish
            U, V, W = _polish(U, V, W, T_target, max_iter=100)
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T_target))
            if err < 1e-9:
                break
    
    # Try snapping to half-integers
    U_snap = _snap_to_half_integers(U)
    V_snap = _snap_to_half_integers(V)
    W_snap = _snap_to_half_integers(W)
    
    if _verify(U_snap, V_snap, W_snap, n, m, p, tol=1e-7):
        return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
    
    # Try rational reconstruction
    U_rat = np.array([[_rational_reconstruct(v) for v in row] for row in U])
    V_rat = np.array([[_rational_reconstruct(v) for v in row] for row in V])
    W_rat = np.array([[_rational_reconstruct(v) for v in row] for row in W])
    
    if _verify(U_rat, V_rat, W_rat, n, m, p, tol=1e-7):
        return (U_rat.tolist(), V_rat.tolist(), W_rat.tolist())
    
    # Also try without snapping (in case it's exact float)
    if _verify(U, V, W, n, m, p, tol=1e-7):
        return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def _try_hankel_svd(rank, n, m, p, seed=42, time_limit=3):
    """Try Hankel/SVD-based approach to find rank decomposition."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    rng = np.random.RandomState(seed)
    
    # Target tensor
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Initialize using SVD of a random flattening
    U = rng.randn(dim_U, rank) * 0.1
    V = rng.randn(dim_V, rank) * 0.1
    W = rng.randn(dim_W, rank) * 0.1
    
    start_time = time.time()
    
    for it in range(100):
        if time.time() > start_time + time_limit:
            break
        
        # Update U
        S_reshaped = np.zeros((rank, dim_V * dim_W))
        for r in range(rank):
            S_reshaped[r, :] = np.outer(V[:, r], W[:, r]).flatten()
        try:
            U_new, _, _, _ = np.linalg.lstsq(S_reshaped.T, T_target.reshape(dim_U, -1).T, rcond=None)
            U_new = U_new.T
        except:
            break
        
        # Update V
        S_UW_reshaped = np.zeros((rank, dim_U * dim_W))
        for r in range(rank):
            S_UW_reshaped[r, :] = np.outer(U_new[:, r], W[:, r]).flatten()
        try:
            V_new, _, _, _ = np.linalg.lstsq(S_UW_reshaped.T, T_target.transpose(1, 0, 2).reshape(dim_V, dim_U * dim_W).T, rcond=None)
            V_new = V_new.T
        except:
            break
        
        # Update W
        S_UV_reshaped = np.zeros((rank, dim_U * dim_V))
        for r in range(rank):
            S_UV_reshaped[r, :] = np.outer(U_new[:, r], V_new[:, r]).flatten()
        try:
            W_new, _, _, _ = np.linalg.lstsq(S_UV_reshaped.T, T_target.transpose(2, 0, 1).reshape(dim_W, dim_U * dim_V).T, rcond=None)
            W_new = W_new.T
        except:
            break
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T_target))
        if err < 1e-8:
            U, V, W = _polish(U, V, W, T_target, max_iter=100)
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T_target))
            if err < 1e-9:
                break
    
    # Try snapping
    U_snap = _snap_to_half_integers(U)
    V_snap = _snap_to_half_integers(V)
    W_snap = _snap_to_half_integers(W)
    
    if _verify(U_snap, V_snap, W_snap, n, m, p, tol=1e-7):
        return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
    
    # Try rational reconstruction
    U_rat = np.array([[_rational_reconstruct(v) for v in row] for row in U])
    V_rat = np.array([[_rational_reconstruct(v) for v in row] for row in V])
    W_rat = np.array([[_rational_reconstruct(v) for v in row] for row in W])
    
    if _verify(U_rat, V_rat, W_rat, n, m, p, tol=1e-7):
        return (U_rat.tolist(), V_rat.tolist(), W_rat.tolist())
    
    if _verify(U, V, W, n, m, p, tol=1e-7):
        return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def construct():
    n, m, p = 2, 3, 4
    
    # Try ranks from 13 to 24
    for rank in range(13, 25):
        if time.time() > _DEADLINE - 15:
            break
        
        # Try ALS with multiple seeds
        max_seeds = 30 if rank <= 15 else 15
        for seed in range(rank * 100, rank * 100 + max_seeds):
            if time.time() > _DEADLINE - 10:
                break
            
            # Try ALS
            res = _try_als(rank, n, m, p, max_iter=150, seed=seed, time_limit=2)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
            
            # Try Hankel/SVD variant
            res = _try_hankel_svd(rank, n, m, p, seed=seed + 1000, time_limit=2)
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
