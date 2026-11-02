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
        Alternating Least Squares.
        """
        R = U.shape[1]
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # For each u, solve min || sum_r U[u,r] (V * W)_r - T[u,:,:] ||
            # (V * W)_r is the outer product V[:,r] W[:,r] flattened?
            # Actually, for fixed u, the equation is:
            # sum_r U[u,r] * (V[:,r] * W[:,r]) = T[u, :, :]
            # Let M_u be a matrix of shape (Nv*Nw, R) where M_u[(v*w), r] = V[v,r] * W[w,r]
            # Then U[u, :] is the solution to M_u @ U[u, :] = T[u, :, :].flatten()
            
            U_new = np.zeros_like(U)
            for u in range(Nu):
                # Construct M_u
                # V: (Nv, R), W: (Nw, R)
                # Outer product of columns?
                # M_u[v*w + w, r] = V[v, r] * W[w, r]
                # This is the element-wise product of V and W broadcast?
                # V[:, :, None] * W[None, :, :] -> (Nv, Nw, R)
                # Flatten to (Nv*Nw, R)
                M_u = (V[:, :, None] * W[None, :, :]).reshape(Nv * Nw, R)
                b_u = T[u, :, :].flatten()
                try:
                    U_new[u, :] = np.linalg.lstsq(M_u, b_u, rcond=None)[0]
                except:
                    U_new[u, :] = U[u, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for v in range(Nv):
                M_v = (U[:, :, None] * W[None, :, :]).reshape(Nu * Nw, R)
                b_v = T[:, v, :].flatten()
                try:
                    V_new[v, :] = np.linalg.lstsq(M_v, b_v, rcond=None)[0]
                except:
                    V_new[v, :] = V[v, :]
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for w in range(Nw):
                M_w = (U[:, :, None] * V[None, :, :]).reshape(Nu * Nv, R)
                b_w = T[:, :, w].flatten()
                try:
                    W_new[w, :] = np.linalg.lstsq(M_w, b_w, rcond=None)[0]
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
    
    for R in range(40, 51):
        if time.time() > deadline - 15:
            break
        found = False
        for seed in range(10):
            if time.time() > deadline - 8:
                break
            result = try_rank(R, seed=seed * 13 + 7)
            if result is not None:
                U_best, V_best, W_best = result
                best_rank = R
                found = True
                break
        if found:
            # If we found a valid decomposition, we can stop if we want to minimize rank
            # But maybe a lower rank is possible?
            # The loop goes 40, 41, 42...
            # So if we find 40, we break.
            break

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
