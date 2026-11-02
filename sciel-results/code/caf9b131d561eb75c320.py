import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

_T_cache = None
def _get_T():
    global _T_cache
    if _T_cache is None:
        _T_cache = _target_tensor()
    return _T_cache

def _verify(U, V, W, tol=1e-8):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = _get_T()
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.abs(recon - T)
    return np.max(diff) < tol

def _als_decompose(T, R, n, m, p, max_iter=100, tol=1e-7, seed=None):
    """
    Alternating least squares to find U (nm x R), V (mp x R), W (pn x R)
    such that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c].
    """
    nm, mp, pn = n*m, m*p, p*n
    
    rng = np.random.RandomState(seed)
    # Random initialization with small scale
    U = rng.randn(nm, R) * 0.1
    V = rng.randn(mp, R) * 0.1
    W = rng.randn(pn, R) * 0.1
    
    for it in range(max_iter):
        # Update U: for each row a, solve least squares
        MW = np.einsum('br,cr->bcr', V, W, optimize=True)  # (mp, pn, R)
        MW_flat = MW.reshape(mp*pn, R)
        MW_T = MW_flat.T  # (R, mp*pn)
        
        for a in range(nm):
            target_a = T[a].reshape(mp*pn)
            X, _, _, _ = np.linalg.lstsq(MW_T, target_a, rcond=None)
            U[a, :] = X
        
        # Update V: for each row b, solve least squares
        UW = np.einsum('ar,cr->acr', U, W, optimize=True)  # (nm, pn, R)
        UW_flat = UW.reshape(nm*pn, R)
        UW_T = UW_flat.T  # (R, nm*pn)
        
        for b in range(mp):
            target_b = T[:, b, :].reshape(nm*pn)
            X, _, _, _ = np.linalg.lstsq(UW_T, target_b, rcond=None)
            V[b, :] = X
        
        # Update W: for each row c, solve least squares
        UV = np.einsum('ar,br->abr', U, V, optimize=True)  # (nm, mp, R)
        UV_flat = UV.reshape(nm*mp, R)
        UV_T = UV_flat.T  # (R, nm*mp)
        
        for c in range(pn):
            target_c = T[:, :, c].reshape(nm*mp)
            X, _, _, _ = np.linalg.lstsq(UV_T, target_c, rcond=None)
            W[c, :] = X
        
        # Check convergence
        recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(recon - T))
        if err < tol:
            break
    
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    err = np.max(np.abs(recon - T))
    if err < 1e-6:
        return U, V, W
    return None

def _construct_schoolbook():
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

def _to_list(M):
    """Convert numpy array to list of lists with integer/half-integer values."""
    M = np.asarray(M, dtype=float)
    out = []
    for row in M:
        out_row = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                out_row.append(int(round(val)))
            elif abs(val * 2 - round(val * 2)) < 1e-9:
                out_row.append(round(val * 2) / 2.0)
            else:
                out_row.append(val)
        out.append(out_row)
    return out

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave margin from 150s limit
    n, m, p = DIMENSIONS
    T = _get_T()
    
    # Start with schoolbook as guaranteed valid
    best_result = _construct_schoolbook()
    best_rank = n * m * p
    
    # Try ranks below schoolbook
    ranks_to_try = [85, 80, 75, 70, 65, 60, 55, 50]
    
    for R in ranks_to_try:
        if time.time() > deadline - 20.0:
            break
        
        n_restarts = 2
        for restart in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            
            result = _als_decompose(T, R, n, m, p, max_iter=80, tol=1e-7, seed=restart*1000+R)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify(U, V, W, tol=1e-8):
                    U_list = _to_list(U)
                    V_list = _to_list(V)
                    W_list = _to_list(W)
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return best_result
