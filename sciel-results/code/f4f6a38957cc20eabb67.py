import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor as a 3D array."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _khatri_rao(A, B):
    """Compute Khatri-Rao product of two matrices A and B."""
    # A is (n, r), B is (m, r)
    # Result is (n*m, r) where result[i*m + j, r] = A[i, r] * B[j, r]
    return np.einsum('ir,jr->ijr', A, B).reshape(A.shape[0] * B.shape[0], -1)

def _als_decompose(T, rank, max_iter=50, tol=1e-10, seed=0):
    """
    Alternating Least Squares to decompose T into U, V, W.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    rng = np.random.RandomState(seed)
    a, b, c = T.shape
    
    # Initialize factors with random values
    U = rng.randn(a, rank) * 0.1
    V = rng.randn(b, rank) * 0.1
    W = rng.randn(c, rank) * 0.1
    
    for it in range(max_iter):
        # Update U
        # T_ac = U @ Khatri(V, W)
        Khatri_VW = _khatri_rao(V, W)
        # T_ac is (a, b*c)
        T_ac = T.reshape(a, b * c)
        try:
            U = np.linalg.lstsq(Khatri_VW, T_ac.T, rcond=None)[0].T
        except:
            pass
            
        # Update V
        # T_bc = V @ Khatri(U, W)
        Khatri_UW = _khatri_rao(U, W)
        T_bc = T.transpose(1, 0, 2).reshape(b, a * c)
        try:
            V = np.linalg.lstsq(Khatri_UW, T_bc.T, rcond=None)[0].T
        except:
            pass
            
        # Update W
        # T_cc = W @ Khatri(U, V)
        Khatri_UV = _khatri_rao(U, V)
        T_cc = T.transpose(2, 0, 1).reshape(c, a * b)
        try:
            W = np.linalg.lstsq(Khatri_UV, T_cc.T, rcond=None)[0].T
        except:
            pass
        
    # Compute residual
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    residual = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
    
    return U, V, W, residual

def _polish(U, V, W, T, max_iter=20):
    """
    Perform a few steps of gradient descent or Newton's method to reduce residual.
    """
    a, b, c = T.shape
    rank = U.shape[1]
    
    # Flatten T
    T_flat = T.flatten()
    
    def residual_func(params):
        U = params[:a*rank].reshape(a, rank)
        V = params[a*rank:a*rank+b*rank].reshape(b, rank)
        W = params[a*rank+b*rank:].reshape(c, rank)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_recon - T).flatten()
    
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Use scipy if available, otherwise simple gradient step
    try:
        from scipy.optimize import least_squares
        res = least_squares(residual_func, params, method='lm', max_nfev=100)
        params = res.x
    except ImportError:
        # Fallback to simple gradient descent if scipy is not available
        # This is slow, but better than nothing
        lr = 0.01
        for _ in range(max_iter):
            r = residual_func(params)
            # Approximate gradient
            # This is very inefficient, so we'll just skip if no scipy
            break
            
    U = params[:a*rank].reshape(a, rank)
    V = params[a*rank:a*rank+b*rank].reshape(b, rank)
    W = params[a*rank+b*rank:].reshape(c, rank)
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    residual = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
    
    return U, V, W, residual

def _try_round(U, V, W, T, step=0.5):
    """
    Try to round U, V, W to nearest multiples of 'step' and check if exact.
    """
    U_r = np.round(U / step) * step
    V_r = np.round(V / step) * step
    W_r = np.round(W / step) * step
    
    T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
    residual = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
    
    if residual < 1e-9:
        return U_r, V_r, W_r, residual
    
    return None, None, None, residual

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Try ranks in increasing order
    # Theoretical lower bound is max(n*m, m*p, p*n) = 27
    # We try ranks from 27 to 54
    target_ranks = list(range(27, rank_schoolbook + 1))
    
    for r in target_ranks:
        if time.time() > deadline - 15:
            break
            
        found = False
        best_local_residual = float('inf')
        best_local_U = best_local_V = best_local_W = None
        
        # Multiple random restarts for each rank
        # Allocate more restarts to lower ranks
        max_restarts = 10 if r < 40 else 5
        
        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            try:
                # Use a different seed for each attempt
                seed = r * 1000 + attempt
                
                # Initialize
                U, V, W, residual = _als_decompose(T, r, max_iter=30, tol=1e-8, seed=seed)
                
                if residual < best_local_residual:
                    best_local_residual = residual
                    best_local_U = U
                    best_local_V = V
                    best_local_W = W
                    
                # If residual is small, try to polish
                if residual < 1e-4:
                    U, V, W, residual = _polish(U, V, W, T)
                    
                    if residual < best_local_residual:
                        best_local_residual = residual
                        best_local_U = U
                        best_local_V = V
                        best_local_W = W
                        
                    # If residual is very small, try rounding
                    if residual < 1e-6:
                        U_r, V_r, W_r, res_r = _try_round(U, V, W, T, step=0.5)
                        if U_r is not None:
                            best_local_residual = res_r
                            best_local_U = U_r
                            best_local_V = V_r
                            best_local_W = W_r
                            found = True
                            break
                            
                        # Also try integer rounding
                        U_r, V_r, W_r, res_r = _try_round(U, V, W, T, step=1.0)
                        if U_r is not None:
                            best_local_residual = res_r
                            best_local_U = U_r
                            best_local_V = V_r
                            best_local_W = W_r
                            found = True
                            break
                            
                        # If not rounded, but residual is extremely small, accept it
                        if residual < 1e-9:
                            found = True
                            break
                            
            except Exception as e:
                continue
        
        if found and best_local_residual < 1e-9:
            best_result = {
                "U": best_local_U.tolist(),
                "V": best_local_V.tolist(),
                "W": best_local_W.tolist()
            }
            best_rank = r
            break
    
    # If no improvement found, return schoolbook
    if best_result is None:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
    else:
        return best_result
