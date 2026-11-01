DIMENSIONS = [3, 3, 3]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=3, tol=1e-6):
    """Verify that U ⊗ V ⊗ W equals the matrix multiplication tensor."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _try_als(rank, n=3, m=3, p=3, max_iters=50, tol=1e-8, seed=0, deadline=None):
    """
    Try to find a rank-`rank` decomposition using ALS.
    Returns (U, V, W) or None.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    rng = np.random.RandomState(seed)
    U = rng.randn(dim_a, rank)
    V = rng.randn(dim_b, rank)
    W = rng.randn(dim_c, rank)
    
    prev_err = float('inf')
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return None

        # Update U
        M = np.einsum('br,cr->bcr', V, W).reshape(dim_b*dim_c, rank)
        for a in range(dim_a):
            U[a] = np.linalg.lstsq(M, T[a].reshape(-1), rcond=None)[0]

        # Update V
        M = np.einsum('ar,cr->acr', U, W).reshape(dim_a*dim_c, rank)
        for b in range(dim_b):
            V[b] = np.linalg.lstsq(M, T[:, b].reshape(-1), rcond=None)[0]

        # Update W
        M = np.einsum('ar,br->abr', U, V).reshape(dim_a*dim_b, rank)
        for c in range(dim_c):
            W[c] = np.linalg.lstsq(M, T[:, :, c].reshape(-1), rcond=None)[0]

        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_approx - T)
        
        if abs(prev_err - err) < tol:
            if err < 1e-4:
                return U, V, W
            return None
        prev_err = err

    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.linalg.norm(T_approx - T)
    if err < 1e-4:
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Try to find a lower rank decomposition using ALS
    # Rank 23 is the known optimal for 3x3.
    # Rank 22-24 are worth a quick try with strict time limits.
    for rank in [22, 23, 24]:
        if time.time() > deadline:
            break
        # Give it a few seconds per rank
        rank_deadline = min(deadline, time.time() + 30)
        # Try a few seeds
        for seed in range(5):
            if time.time() > rank_deadline:
                break
            res = _try_als(rank, n, m, p, max_iters=100, tol=1e-9, seed=seed, deadline=rank_deadline)
            if res is not None:
                U, V, W = res
                if _verify(U, V, W, n, m, p, tol=1e-5):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # 2. Fallback to Schoolbook (Rank 27)
    rank = 27
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
