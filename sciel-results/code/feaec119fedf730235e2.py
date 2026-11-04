import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    T = np.zeros((n*m, m*p, p*n))
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                nz_indices.append((a, b, c))
                
    # Precompute shapes
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    size_U_flat = dim_V * dim_W
    size_V_flat = dim_U * dim_W
    size_W_flat = dim_U * dim_V

    best_result = None
    
    for rank in range(39, 24, -1):
        if time.time() > deadline - 20:
            break
            
        trials = 20 if rank > 35 else 50
        
        for trial in range(trials):
            if time.time() > deadline - 10:
                break
                
            U = np.random.randn(dim_U, rank) * 0.5
            V = np.random.randn(dim_V, rank) * 0.5
            W = np.random.randn(dim_W, rank) * 0.5
            
            for it in range(300):
                if time.time() > deadline - 5:
                    break
                    
                # Build X_U (size_U_flat x rank)
                # X_U[:, r] = (V[:, r] * W[:, r]).flatten() ? No.
                # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed a: T[a, :, :] = sum_r U[a,r] (V[:, r] W[:, r]^T)
                # So X_U[:, r] = (V[:, r] * W[:, r]^T).flatten()
                # Wait, V[:, r] is dim_V x 1. W[:, r] is dim_W x 1.
                # Outer product is dim_V x dim_W.
                X_U = np.zeros((size_U_flat, rank))
                for r in range(rank):
                    X_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                    
                # Update U
                for a in range(dim_U):
                    target = T[a, :, :].flatten()
                    try:
                        U[a, :] = np.linalg.lstsq(X_U, target, rcond=None)[0]
                    except:
                        pass
                        
                # Build X_V (size_V_flat x rank)
                # For fixed b: T[:, b, :] = sum_r V[b,r] (U[:, r] * W[:, r])
                # X_V[:, r] = (U[:, r] * W[:, r]).flatten()
                X_V = np.zeros((size_V_flat, rank))
                for r in range(rank):
                    X_V[:, r] = (U[:, r] * W[:, r]).flatten()
                    
                # Update V
                for b in range(dim_V):
                    target = T[:, b, :].flatten()
                    try:
                        V[b, :] = np.linalg.lstsq(X_V, target, rcond=None)[0]
                    except:
                        pass
                        
                # Build X_W (size_W_flat x rank)
                # For fixed c: T[:, :, c] = sum_r W[c,r] (U[:, r] * V[:, r])
                # X_W[:, r] = (U[:, r] * V[:, r]).flatten()
                X_W = np.zeros((size_W_flat, rank))
                for r in range(rank):
                    X_W[:, r] = (U[:, r] * V[:, r]).flatten()
                    
                # Update W
                for c in range(dim_W):
                    target = T[:, :, c].flatten()
                    try:
                        W[c, :] = np.linalg.lstsq(X_W, target, rcond=None)[0]
                    except:
                        pass
                        
                # Check error
                # Fast check on non-zero entries
                max_err = 0.0
                for a, b, c in nz_indices:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
                        
                if max_err < 1e-6:
                    # Full check
                    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                    err_full = np.max(np.abs(T_recon - T))
                    if err_full < 1e-6:
                        # Round to half-integers
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        
                        T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                        err_r = np.max(np.abs(T_recon_r - T))
                        
                        if err_r < 1e-9:
                            return {
                                "U": U_r.tolist(),
                                "V": V_r.tolist(),
                                "W": W_r.tolist()
                            }
                        
                        # If rounding to 0.5 failed, try rounding to 0.25?
                        # Task says integer/half-integer.
                        # If it fails, maybe we need more iterations.
                        # Let's continue ALS for a bit more if we are close.
                        pass
    
    # Fallback
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

