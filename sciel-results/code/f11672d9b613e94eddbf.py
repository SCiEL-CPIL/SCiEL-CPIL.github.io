import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor as a 3D array."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _als_decompose(T, rank, max_iter=200, tol=1e-10, seed=0):
    """
    Alternating Least Squares to decompose T into U, V, W.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    rng = np.random.RandomState(seed)
    a, b, c = T.shape
    
    # Initialize factors
    U = rng.randn(a, rank) * 0.1
    V = rng.randn(b, rank) * 0.1
    W = rng.randn(c, rank) * 0.1
    
    for it in range(max_iter):
        # Update U: for each a, solve for U[a,:] given V, W
        # T[a,:,:] = sum_r U[a,r] * V[:,r] * W[:,r]
        # This is a linear system in U[a,:]
        # Let's vectorize: reshape T to (a, b*c)
        T_ac = T.reshape(a, b * c)
        
        # Build design matrix: for each a, we need to solve
        # T_ac[a, :] = U[a, :] @ (V ⊙ W) where ⊙ is Khatri-Rao product
        # (V ⊙ W)[:, r] = V[:, r] * W[:, r]
        KW = (V * W).T  # shape (rank, b*c) -> actually V.T * W.T element-wise
        # More carefully: Khatri-Rao of V and W
        # (V ⊙ W) has columns v_r * w_r
        VRW = (V.T * W.T)  # (rank, b) * (rank, c) -> need to be careful
        # Actually: for each r, the column is V[:,r] * W[:,r] (element-wise in the b*c space)
        # So KhatriRao(V, W) = np.einsum('br,cr->bcr', V, W).reshape(-1, rank).T
        # Let me just compute it directly
        Khatri = np.einsum('br,cr->bcr', V, W).reshape(b * c, rank).T  # (rank, b*c)
        
        # Solve U: T_ac = U @ Khatri
        # U = T_ac @ Khatri^+ 
        try:
            U_new = np.linalg.lstsq(Khatri, T_ac.T, rcond=None)[0].T
        except:
            U_new = U
        
        # Update V: similar
        T_bc = T.transpose(1, 0, 2).reshape(b, a * c)
        Khatri2 = np.einsum('ar,cr->acr', U_new, W).reshape(a * c, rank).T
        try:
            V_new = np.linalg.lstsq(Khatri2, T_bc.T, rcond=None)[0].T
        except:
            V_new = V
        
        # Update W: similar
        T_cc = T.transpose(2, 0, 1).reshape(c, a * b)
        Khatri3 = np.einsum('ar,br->abr', U_new, V_new).reshape(a * b, rank).T
        try:
            W_new = np.linalg.lstsq(Khatri3, T_cc.T, rcond=None)[0].T
        except:
            W_new = W
        
        # Check convergence
        U_diff = np.linalg.norm(U_new - U)
        V_diff = np.linalg.norm(V_new - V)
        W_diff = np.linalg.norm(W_new - W)
        
        U, V, W = U_new, V_new, W_new
        
        if U_diff + V_diff + W_diff < tol:
            break
    
    # Compute residual
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    residual = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
    
    return U, V, W, residual

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    # Deadline guard
    start_time = time.time()
    deadline = 135.0  # Leave 15s margin
    
    best_result = None
    best_rank = rank_schoolbook
    best_residual = float('inf')
    
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
    
    # Try ranks in increasing order
    # For 2x3x9, theoretical lower bound is around 30-40
    target_ranks = list(range(28, 55))
    
    for r in target_ranks:
        if time.time() > deadline - 10:
            break
        
        found = False
        best_local_residual = float('inf')
        best_local_U = best_local_V = best_local_W = None
        
        # Multiple random restarts for each rank
        max_restarts = 5
        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            try:
                U, V, W, residual = _als_decompose(T, r, max_iter=100, tol=1e-12, seed=attempt * 100)
                
                if residual < best_local_residual:
                    best_local_residual = residual
                    best_local_U = U
                    best_local_V = V
                    best_local_W = W
                    
                # If residual is very small, we have a good solution
                if residual < 1e-8:
                    found = True
                    break
                    
            except Exception as e:
                continue
        
        if found and best_local_residual < 1e-8:
            best_result = {
                "U": best_local_U.tolist(),
                "V": best_local_V.tolist(),
                "W": best_local_W.tolist()
            }
            best_rank = r
            best_residual = best_local_residual
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
