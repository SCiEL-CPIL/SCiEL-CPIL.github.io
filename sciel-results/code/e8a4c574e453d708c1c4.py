import numpy as np
import time

DIMENSIONS = [2, 2, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (14, rank)
    # W: (p*n, rank) = (14, rank)
    
    # Build target tensor T of shape (4, 14, 14)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 28)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        rank = U.shape[1]
        # Compute sum_r U_r ⊗ V_r ⊗ W_r
        # T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=tol)
    
    def als_update_U(U, V, W, T):
        """Update U using ALS. For each a, solve: T[a,:,:] = sum_r U[a,r] (V[:,r] ⊗ W[:,r])"""
        r = U.shape[1]
        # M_a[:, r] = V[:, r] ⊗ W[:, r]
        # For each a: T[a].reshape(-1) = M_a @ U[a,:]
        U_new = np.zeros_like(U)
        for a in range(U.shape[0]):
            # Build M_a: shape (m*p * p*n, r)
            M_a = np.zeros((m*p * p*n, r))
            for rr in range(r):
                M_a[:, rr] = np.kron(V[:, rr], W[:, rr])
            # Solve least squares
            try:
                U_new[a, :] = np.linalg.lstsq(M_a, T[a].reshape(-1), rcond=None)[0]
            except:
                U_new[a, :] = U[a, :]
        return U_new
    
    def als_update_V(U, V, W, T):
        """Update V using ALS. For each b, solve: T[:,b,:] = sum_r V[b,r] (U[:,r] ⊗ W[:,r])"""
        r = V.shape[1]
        V_new = np.zeros_like(V)
        for b in range(V.shape[0]):
            # Build M_b: shape (n*m * p*n, r)
            M_b = np.zeros((n*m * p*n, r))
            for rr in range(r):
                M_b[:, rr] = np.kron(U[:, rr], W[:, rr])
            # Solve least squares
            try:
                V_new[b, :] = np.linalg.lstsq(M_b, T[:, b, :].reshape(-1), rcond=None)[0]
            except:
                V_new[b, :] = V[b, :]
        return V_new
    
    def als_update_W(U, V, W, T):
        """Update W using ALS. For each c, solve: T[:,:,c] = sum_r W[c,r] (U[:,r] ⊗ V[:,r])"""
        r = W.shape[1]
        W_new = np.zeros_like(W)
        for c in range(W.shape[0]):
            # Build M_c: shape (n*m * m*p, r)
            M_c = np.zeros((n*m * m*p, r))
            for rr in range(r):
                M_c[:, rr] = np.kron(U[:, rr], V[:, rr])
            # Solve least squares
            try:
                W_new[c, :] = np.linalg.lstsq(M_c, T[:, :, c].reshape(-1), rcond=None)[0]
            except:
                W_new[c, :] = W[c, :]
        return W_new
    
    def try_rank(r, max_iters=200, n_restarts=10):
        """Try to find a rank-r decomposition using ALS."""
        nonlocal best_U, best_V, best_W, best_rank
        if r >= best_rank:
            return False
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return False
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            prev_loss = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    return False
                
                # ALS updates
                U = als_update_U(U, V, W, T)
                V = als_update_V(U, V, W, T)
                W = als_update_W(U, V, W, T)
                
                # Check convergence
                T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                loss = np.sum((T_rec - T)**2)
                
                if loss < 1e-18:
                    if check_decomposition(U, V, W):
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = r
                        return True
                    break
                
                if abs(prev_loss - loss) < 1e-12 * max(1, prev_loss):
                    break
                prev_loss = loss
            
            # Final check
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_rec, T, atol=1e-9):
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = r
                return True
        
        return False
    
    # Try ranks from low to high
    for r in range(14, best_rank):
        if time.time() > deadline:
            break
        if try_rank(r):
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
