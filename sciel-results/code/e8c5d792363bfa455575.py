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

def _als_decompose(T, R, n_iter=200, seed=42):
    """
    Alternating Least Squares for tensor decomposition.
    T: shape (A, B, C)
    R: rank
    Returns U, V, W or None if it fails to converge.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(A, R) * 0.1
    V = rng.randn(B, R) * 0.1
    W = rng.randn(C, R) * 0.1
    
    for it in range(n_iter):
        # Solve for U
        A_U = np.zeros((B * C, R))
        for r in range(R):
            A_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
        
        for a in range(A):
            b_a = T[a, :, :].flatten()
            u, _, _, _ = np.linalg.lstsq(A_U, b_a, rcond=None)
            U[a, :] = u
            
        # Solve for V
        A_V = np.zeros((A * C, R))
        for r in range(R):
            A_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
        for b in range(B):
            b_b = T[:, b, :].flatten()
            v, _, _, _ = np.linalg.lstsq(A_V, b_b, rcond=None)
            V[b, :] = v
            
        # Solve for W
        A_W = np.zeros((A * B, R))
        for r in range(R):
            A_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
        for c in range(C):
            b_c = T[:, :, c].flatten()
            w, _, _, _ = np.linalg.lstsq(A_W, b_c, rcond=None)
            W[c, :] = w
            
        # Check convergence
        if it % 10 == 0:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            if np.max(np.abs(T_recon - T)) < 1e-6:
                break
                
    # Final check
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_recon - T)) < 1e-5:
        return U, V, W
    return None

def _try_rank_als(rank, n, m, p, max_time=5, seed=42):
    """Try to find a rank-rank decomposition via ALS."""
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

    res = _als_decompose(T_target, rank, n_iter=100, seed=seed)
    
    if res is not None:
        U, V, W = res
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
            
    return None

def construct():
    n, m, p = 2, 3, 4
    
    # Try ranks from 22 to 24
    # 22 is the conjectured minimum, 24 is schoolbook
    for rank in [22, 23, 24]:
        if time.time() > _DEADLINE - 10:
            break
            
        # Try multiple seeds
        # Increase seeds for lower ranks to improve chance of finding the global min
        num_seeds = 50 if rank < 24 else 10
        for seed in range(num_seeds):
            if time.time() > _DEADLINE - 2:
                break
            res = _try_rank_als(rank, n, m, p, seed=seed)
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
