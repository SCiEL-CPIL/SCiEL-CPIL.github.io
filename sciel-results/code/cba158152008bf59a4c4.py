import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 2, 5, 5
    Nu = n * m  # 10
    Nv = m * p  # 25
    Nw = p * n  # 10
    
    # Build the target tensor T as a dense array for simplicity, 
    # since dimensions are small (10x25x10 = 2500 entries).
    T = np.zeros((Nu, Nv, Nw))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    def compute_residual(U, V, W):
        # Compute sum_r U[u,r] V[v,r] W[w,r]
        # Using einsum
        # U: (Nu, R), V: (Nv, R), W: (Nw, R)
        # Residual: (Nu, Nv, Nw)
        # R_uvw = sum_r U[u,r] V[v,r] W[w,r]
        # This is equivalent to tensor product
        # We can do:
        # Temp = np.einsum('ir,vr->iv', U, V) # (Nu, Nv, R)
        # Res = np.einsum('ivr,wr->uvw', Temp, W)
        Temp = np.einsum('ir,vr->ivr', U, V)
        Res = np.einsum('ivr,wr->uvw', Temp, W)
        return Res - T

    def max_abs_residual(U, V, W):
        res = compute_residual(U, V, W)
        return np.max(np.abs(res))

    def als_step(U, V, W, T, max_iter=100, tol=1e-10):
        """
        Optimized Alternating Least Squares.
        Precomputes the coefficient matrices for each factor update.
        """
        R = U.shape[1]
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # Coefficient matrix M_U is constant for all u
            # M_U[(v, w), r] = V[v, r] * W[w, r]
            M_U = (V[:, :, None] * W[None, :, :]).reshape(Nv * Nw, R)
            
            U_new = np.zeros_like(U)
            for u in range(Nu):
                b_u = T[u, :, :].flatten()
                try:
                    # Solve M_U.T @ U_new[u, :].T = b_u
                    U_new[u, :] = np.linalg.lstsq(M_U.T, b_u, rcond=None)[0]
                except:
                    U_new[u, :] = U[u, :]
            U = U_new
            
            # Update V
            # Coefficient matrix M_V is constant for all v
            # M_V[(u, w), r] = U[u, r] * W[w, r]
            M_V = (U[:, :, None] * W[None, :, :]).reshape(Nu * Nw, R)
            
            V_new = np.zeros_like(V)
            for v in range(Nv):
                b_v = T[:, v, :].flatten()
                try:
                    # Solve M_V.T @ V_new[v, :].T = b_v
                    V_new[v, :] = np.linalg.lstsq(M_V.T, b_v, rcond=None)[0]
                except:
                    V_new[v, :] = V[v, :]
            V = V_new
            
            # Update W
            # Coefficient matrix M_W is constant for all w
            # M_W[(u, v), r] = U[u, r] * V[v, r]
            M_W = (U[:, :, None] * V[None, :, :]).reshape(Nu * Nv, R)
            
            W_new = np.zeros_like(W)
            for w in range(Nw):
                b_w = T[:, :, w].flatten()
                try:
                    # Solve M_W.T @ W_new[w, :].T = b_w
                    W_new[w, :] = np.linalg.lstsq(M_W.T, b_w, rcond=None)[0]
                except:
                    W_new[w, :] = W[w, :]
            W = W_new
            
            # Check convergence
            err = max_abs_residual(U, V, W)
            if err < tol:
                break
                
        return U, V, W

    def try_rank(R, seed=42):
        if time.time() > deadline - 10:
            return None
            
        np.random.seed(seed)
        
        best_result = None
        best_err = float('inf')
        
        # Try a few initializations
        for init_iter in range(5):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(Nu, R)
            V = np.random.randn(Nv, R)
            W = np.random.randn(Nw, R)
            
            # Normalize to prevent divergence
            for r in range(R):
                U[:, r] /= (np.max(np.abs(U[:, r])) + 1e-10)
                V[:, r] /= (np.max(np.abs(V[:, r])) + 1e-10)
                W[:, r] /= (np.max(np.abs(W[:, r])) + 1e-10)
            
            U, V, W = als_step(U, V, W, T, max_iter=50, tol=1e-9)
            
            err = max_abs_residual(U, V, W)
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-8:
                return U, V, W
                
        if best_result is not None and best_err < 1e-8:
            return best_result
        return None

    # Try ranks from 40 to 50
    # Known lower bounds for 2x5x5 are likely around 40-42
    best_rank = 50
    U_best, V_best, W_best = None, None, None
    
    # Initialize with schoolbook rank 50 as fallback
    U_fb = np.zeros((Nu, 50))
    V_fb = np.zeros((Nv, 50))
    W_fb = np.zeros((Nw, 50))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    U_best, V_best, W_best = U_fb, V_fb, W_fb
    
    # Known lower bound is likely 40. Try 40 first with more seeds.
    ranks_to_try = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        found = False
        # Use more seeds for lower ranks
        num_seeds = 20 if R <= 42 else 10
        for seed in range(num_seeds):
            if time.time() > deadline - 8:
                break
            result = try_rank(R, seed=seed * 13 + 7)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = R
                found = True
                break
        if found:
            break

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
