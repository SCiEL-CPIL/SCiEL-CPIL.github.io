import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _construct_target_tensor(n, m, p):
    """
    Construct the dense target tensor for matrix multiplication.
    Shape: (n*m, m*p, p*n)
    Entry (I, J, K) is 1 if I=(i,j), J=(j,k), K=(k,i) are consistent.
    """
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, T_target, tol=1e-9):
    """
    Verify if U, V, W decompose T_target.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    # T_calc[I, J, K] = sum_c U[I, c] * V[J, c] * W[K, c]
    # Using einsum
    T_calc = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
    return np.allclose(T_calc, T_target, atol=tol, rtol=0.0)

def _round_to_half(X):
    """Round matrix to nearest 0.5."""
    return np.round(X * 2.0) / 2.0

def _als_decompose(n, m, p, rank, T_target, max_iter=20, tol=1e-8, seed=42):
    """
    Alternating Least Squares for tensor decomposition.
    """
    np.random.seed(seed)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Initialize
    U = np.random.randn(nm, rank) * 0.1
    V = np.random.randn(mp, rank) * 0.1
    W = np.random.randn(pn, rank) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iter in range(max_iter):
        # Update U
        # For each I, solve A_I u_I = t_I
        # A_I has shape (mp*pn, rank)
        # t_I has shape (mp*pn,)
        
        # We can vectorize this if we structure it carefully, but for small n,m,p, loop is fine.
        # However, constructing A_I in a loop is slow.
        # Let's try to do it efficiently.
        
        # Reshape T_target to (nm, mp*pn)
        T_flat = T_target.reshape(nm, mp * pn)
        
        # Precompute V_flat and W_flat? No, we need outer product per c.
        
        # Loop over I
        U_new = np.zeros_like(U)
        for i in range(nm):
            # Construct A_i
            # A_i[(J, K), c] = V[J, c] * W[K, c]
            # V is (mp, rank), W is (pn, rank)
            # Outer product of V and W along the rank dimension?
            # V[:, :, None] * W[None, :, :] -> (mp, pn, rank)
            # Reshape to (mp*pn, rank)
            A_i = (V[:, :, None] * W[None, :, :]).reshape(mp * pn, rank)
            t_i = T_flat[i]
            
            # Solve A_i u = t_i
            try:
                sol, residuals, rank_sol, sv = np.linalg.lstsq(A_i, t_i, rcond=None)
                U_new[i] = sol
            except np.linalg.LinAlgError:
                U_new[i] = U[i] # Keep old value if error
        
        U = U_new
        
        # Update V
        # For each J, solve A_j v_j = t_j
        # A_j[(I, K), c] = U[I, c] * W[K, c]
        T_flat_v = T_target.transpose(1, 0, 2).reshape(mp, nm * pn)
        
        V_new = np.zeros_like(V)
        for j in range(mp):
            A_j = (U[:, :, None] * W[None, :, :]).reshape(nm * pn, rank)
            t_j = T_flat_v[j]
            
            try:
                sol, residuals, rank_sol, sv = np.linalg.lstsq(A_j, t_j, rcond=None)
                V_new[j] = sol
            except np.linalg.LinAlgError:
                V_new[j] = V[j]
        
        V = V_new
        
        # Update W
        # For each K, solve A_k w_k = t_k
        # A_k[(I, J), c] = U[I, c] * V[J, c]
        T_flat_w = T_target.transpose(2, 0, 1).reshape(pn, nm * mp)
        
        W_new = np.zeros_like(W)
        for k in range(pn):
            A_k = (U[:, :, None] * V[None, :, :]).reshape(nm * mp, rank)
            t_k = T_flat_w[k]
            
            try:
                sol, residuals, rank_sol, sv = np.linalg.lstsq(A_k, t_k, rcond=None)
                W_new[k] = sol
            except np.linalg.LinAlgError:
                W_new[k] = W[k]
        
        W = W_new
        
        # Calculate error
        T_calc = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
        err = np.linalg.norm(T_calc - T_target)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < tol:
            break
            
    return best_U, best_V, best_W, best_err

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s safety margin
    
    T_target = _construct_target_tensor(n, m, p)
    
    best_rank = n * m * p
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound for 2x5x6 is likely around 55
    # We try 55, 56, 57, 58, 59, 60
    for target_rank in range(55, best_rank + 1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
                
            try:
                U, V, W, err = _als_decompose(n, m, p, target_rank, T_target, max_iter=30, seed=seed)
                
                # Check if error is small enough to attempt rounding
                if err < 1e-6:
                    U_r = _round_to_half(U)
                    V_r = _round_to_half(V)
                    W_r = _round_to_half(W)
                    
                    if _verify_decomposition(U_r, V_r, W_r, n, m, p, T_target, tol=1e-9):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        
            except Exception:
                continue
                
        # If we found a valid decomposition for this rank (even if not rounded yet, but we only return rounded), 
        # we would have returned. 
        # If we didn't find a rounded solution, we continue to next rank? 
        # Actually, if ALS converged to low error but rounding failed, it's possible a different solution exists.
        # But if no rank works, we fallback to schoolbook.
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
