import numpy as np
import time

def _construct_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, T):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.abs(recon - T)
    return np.max(diff) < 1e-8

def _round_half_int(M, tol=1e-6):
    """Round a matrix to nearest half-integers if possible."""
    M2 = 2.0 * np.asarray(M, dtype=float)
    Mr = np.round(M2)
    if np.max(np.abs(M2 - Mr)) < tol:
        return Mr / 2.0, True
    return None, False

def _als_decompose(T, R, n, m, p, max_iter=50, tol=1e-7, seed=None):
    """
    Alternating least squares to find U (nm x R), V (mp x R), W (pn x R)
    such that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c].
    """
    nm, mp, pn = n*m, m*p, p*n
    
    if seed is not None:
        np.random.seed(seed)
    
    # Random initialization
    U = np.random.randn(nm, R) * 0.5
    V = np.random.randn(mp, R) * 0.5
    W = np.random.randn(pn, R) * 0.5

    for it in range(max_iter):
        # Update U
        MW = np.einsum('br,cr->bcr', V, W, optimize=True)  # (mp, pn, R)
        MW_flat = MW.reshape(mp*pn, R)
        
        for a in range(nm):
            target_a = T[a].reshape(mp*pn)
            X, residuals, rank, sv = np.linalg.lstsq(MW_flat.T, target_a, rcond=None)
            U[a, :] = X
        
        # Update V
        UW = np.einsum('ar,cr->acr', U, W, optimize=True)  # (nm, pn, R)
        UW_flat = UW.reshape(nm*pn, R)
        
        for b in range(mp):
            target_b = T[:, b, :].reshape(nm*pn)
            X, residuals, rank, sv = np.linalg.lstsq(UW_flat.T, target_b, rcond=None)
            V[b, :] = X
        
        # Update W
        UV = np.einsum('ar,br->abr', U, V, optimize=True)  # (nm, mp, R)
        UV_flat = UV.reshape(nm*mp, R)
        
        for c in range(pn):
            target_c = T[:, :, c].reshape(nm*mp)
            X, residuals, rank, sv = np.linalg.lstsq(UV_flat.T, target_c, rcond=None)
            W[c, :] = X
        
        # Check convergence
        recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(recon - T))
        if err < tol:
            break
    
    recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    err = np.max(np.abs(recon - T))
    if err < 1e-6:
        # Try to round to half-integers
        U_r, ok1 = _round_half_int(U, tol=1e-4)
        if ok1:
            V_r, ok2 = _round_half_int(V, tol=1e-4)
            if ok2:
                W_r, ok3 = _round_half_int(W, tol=1e-4)
                if ok3:
                    if _verify(U_r, V_r, W_r, T):
                        return U_r, V_r, W_r
    return None

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 125
    
    T = _target_tensor(n, m, p)
    
    # The trivial decomposition is rank 90.
    # We attempt to find a lower rank decomposition via ALS.
    # If successful, we return it. Otherwise, we fall back to the trivial one.
    
    # Try a few ranks below 90, but with a strict time limit.
    # Given the difficulty, we prioritize returning a valid solution quickly.
    # We will try rank 89, 88, etc. but with very few iterations/restarts.
    
    # For this specific task, the "score" is the rank.
    # The seed returns 90.
    # If we can find 89, we win.
    
    # Let's try a quick search for rank < 90.
    # If we find one, great. If not, return 90.
    
    # To be safe and fast, we only try a very limited number of ALS runs.
    # The ALS in the inspiration was quite slow.
    
    # Let's try rank 89 with 1 restart and 20 iterations.
    if time.time() < deadline - 30:
        result = _als_decompose(T, 89, n, m, p, max_iter=20, tol=1e-7, seed=42)
        if result is not None:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback to trivial
    result = _construct_trivial(n, m, p)
    return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
