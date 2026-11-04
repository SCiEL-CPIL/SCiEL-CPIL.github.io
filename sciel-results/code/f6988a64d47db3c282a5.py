import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    nm = n * m
    mp = m * p
    pn = p * n

    # Build target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        # Check half-integer constraint
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if not (np.allclose(U, U_r, atol=1e-6) and 
                np.allclose(V, V_r, atol=1e-6) and 
                np.allclose(W, W_r, atol=1e-6)):
            return False
            
        # Check exact decomposition
        # Reconstruct T_rec
        # T_rec[a, b, c] = sum_r U_r[a, r] V_r[b, r] W_r[c, r]
        # Use einsum for speed
        T_rec = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        
        if not np.allclose(T_rec, T, atol=tol):
            return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    fallback = schoolbook()

    def try_als(rank, max_iters=200, num_restarts=5):
        if time.time() > deadline:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
                
            np.random.seed(42 + restart)
            # Sparse initialization to help convergence to sparse solutions
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Threshold to sparsify
            sparsity_threshold = 0.5
            U[np.abs(U) < sparsity_threshold] = 0
            V[np.abs(V) < sparsity_threshold] = 0
            W[np.abs(W) < sparsity_threshold] = 0
            
            prev_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                # A[b*pn+c, r] = V[b, r] * W[c, r]
                # t = T[a, :, :].reshape(-1)
                # This is expensive to build A for each a.
                # Let's try a simpler update: 
                # U[a, r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r] / sum_{b,c} (V[b,r] W[c,r])^2 ?
                # This is gradient descent.
                
                # Let's stick to lstsq but optimize the loop.
                # Pre-compute VW_outer = (V[:, None, :] * W[None, :, :]).reshape(-1, rank)
                VW_outer = (V[:, None, :] * W[None, :, :]).reshape(-1, rank)
                UW_outer = (U[:, None, :] * W[None, :, :]).reshape(-1, rank)
                UV_outer = (U[:, None, :] * V[None, :, :]).reshape(-1, rank)
                
                for a in range(nm):
                    t = T[a, :, :].reshape(-1)
                    U[a, :], residuals, rank_val, sv = np.linalg.lstsq(VW_outer, t, rcond=None)
                
                VW_outer = (V[:, None, :] * W[None, :, :]).reshape(-1, rank)
                for b in range(mp):
                    t = T[:, b, :].reshape(-1)
                    V[b, :], residuals, rank_val, sv = np.linalg.lstsq(UW_outer, t, rcond=None)
                    
                UW_outer = (U[:, None, :] * W[None, :, :]).reshape(-1, rank)
                for c in range(pn):
                    t = T[:, :, c].reshape(-1)
                    W[c, :], residuals, rank_val, sv = np.linalg.lstsq(UV_outer, t, rcond=None)
                    
                # Sparsify
                U[np.abs(U) < 0.1] = 0
                V[np.abs(V) < 0.1] = 0
                W[np.abs(W) < 0.1] = 0
                
                # Check error
                T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_rec - T)
                
                if err < 1e-6:
                    if verify(U, V, W):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        
                if err > prev_err * 1.1:
                    break
                prev_err = err

        return None

    best_result = None
    # Try ranks from 80 to 119
    for r in range(80, 120):
        if time.time() > deadline - 10:
            break
            
        result = try_als(r, max_iters=50, num_restarts=3)
        if result is not None:
            best_result = result
            break
            
    if best_result is None:
        best_result = fallback
        
    return best_result
