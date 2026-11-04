import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, 40s margin for safety
    
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # Try ranks from low to high
    # Known upper bound is 24.
    # Try 20, 21, 22, 23
    ranks_to_try = [20, 21, 22, 23]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try a few restarts for this rank
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with random values
            # Shapes: (n, m, r), (m, p, r), (p, n, r)
            U = np.random.randn(n, m, r)
            V = np.random.randn(m, p, r)
            W = np.random.randn(p, n, r)
            
            max_iter = 100
            tol = 1e-12
            
            for it in range(max_iter):
                prev_U = U.copy()
                prev_V = V.copy()
                prev_W = W.copy()
                
                # Update U
                # U[i, j, r] = sum_k T[i, j, k] V[j, k, r] W[k, i, r] / sum_k V[j, k, r]^2 W[k, i, r]^2
                # T is all 1s.
                # Numerator: sum_k V[j, k, r] W[k, i, r] = (V_r @ W_r.T)[j, i].T
                # Denominator: sum_k V[j, k, r]^2 W[k, i, r]^2
                # Let V2 = V**2, W2 = W**2
                # D[i, j, r] = sum_k V2[j, k, r] W2[k, i, r]
                
                U_new = np.zeros_like(U)
                V2 = V ** 2
                W2 = W ** 2
                
                for r_idx in range(r):
                    # V_r shape (m, p), W_r shape (p, n)
                    # Num shape (n, m)
                    num = (V[:, :, r_idx] @ W[:, :, r_idx].T).T
                    # Den shape (n, m)
                    # D[i, j] = sum_k V2[j, k] W2[k, i]
                    # V2_r shape (m, p), W2_r shape (p, n)
                    den = (V2[:, :, r_idx] @ W2[:, :, r_idx].T).T
                    mask = den > 1e-12
                    U_new[:, :, r_idx] = np.where(mask, num / np.where(mask, den, 1), 0)
                U = U_new
                
                # Update V
                V_new = np.zeros_like(V)
                U2 = U ** 2
                for r_idx in range(r):
                    # U_r shape (n, m), W_r shape (p, n)
                    # Num shape (m, p)
                    # N[j, k] = sum_i U[i, j] W[k, i] = (U_r.T @ W_r)[j, k]
                    num = U[:, :, r_idx].T @ W[:, :, r_idx]
                    # Den shape (m, p)
                    # D[j, k] = sum_i U2[i, j] W2[k, i]
                    # U2_r shape (n, m), W2_r shape (p, n)
                    den = U2[:, :, r_idx].T @ W2[:, :, r_idx]
                    mask = den > 1e-12
                    V_new[:, :, r_idx] = np.where(mask, num / np.where(mask, den, 1), 0)
                V = V_new
                
                # Update W
                W_new = np.zeros_like(W)
                V2 = V ** 2
                for r_idx in range(r):
                    # U_r shape (n, m), V_r shape (m, p)
                    # Num shape (p, n)
                    # N[k, i] = sum_j U[i, j] V[j, k] = (U_r @ V_r)[i, k].T
                    num = (U[:, :, r_idx] @ V[:, :, r_idx]).T
                    # Den shape (p, n)
                    # D[k, i] = sum_j U2[i, j] V2[j, k]
                    # U2_r shape (n, m), V2_r shape (m, p)
                    den = (U2[:, :, r_idx] @ V2[:, :, r_idx]).T
                    mask = den > 1e-12
                    W_new[:, :, r_idx] = np.where(mask, num / np.where(mask, den, 1), 0)
                W = W_new
                
                # Check convergence
                if np.linalg.norm(U - prev_U) + np.linalg.norm(V - prev_V) + np.linalg.norm(W - prev_W) < tol:
                    break
                    
            # Check error
            # Reconstruct T_rec
            # T_rec[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
            T_rec = np.zeros((n, m, p))
            for r_idx in range(r):
                # Term = U[:,:,r] (n,m) * V[:,:,r] (m,p) * W[:,:,r] (p,n)
                # This is not a standard element-wise product of 3 matrices.
                # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Let's compute this efficiently.
                # For fixed r, we have tensors.
                # We can use einsum or loops.
                # Since n, m, p are small, loops are fine.
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_rec[i, j, k] += U[i, j, r_idx] * V[j, k, r_idx] * W[k, i, r_idx]
                            
            error = np.linalg.norm(T - T_rec)
            
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            T_snap = np.zeros((n, m, p))
            for r_idx in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_snap[i, j, k] += U_snap[i, j, r_idx] * V_snap[j, k, r_idx] * W_snap[k, i, r_idx]
                            
            snap_error = np.linalg.norm(T - T_snap)
            
            if snap_error < 1e-9:
                # Found a valid solution
                # Convert to lists
                U_list = U_snap.tolist()
                V_list = V_snap.tolist()
                W_list = W_snap.tolist()
                
                # Reshape to required formats:
                # U: (n*m, rank)
                # V: (m*p, rank)
                # W: (p*n, rank)
                
                U_flat = np.zeros((n * m, r))
                V_flat = np.zeros((m * p, r))
                W_flat = np.zeros((p * n, r))
                
                for i in range(n):
                    for j in range(m):
                        U_flat[i * m + j] = U_snap[i, j]
                        
                for j in range(m):
                    for k in range(p):
                        V_flat[j * p + k] = V_snap[j, k]
                        
                for k in range(p):
                    for i in range(n):
                        W_flat[k * n + i] = W_snap[k, i]
                        
                return {
                    "U": U_flat.tolist(),
                    "V": V_flat.tolist(),
                    "W": W_flat.tolist()
                }
                
    # Fallback to schoolbook
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
