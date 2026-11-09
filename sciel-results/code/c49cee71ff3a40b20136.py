import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety and final checks
    
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build target tensor T
    # T is a 3D array of shape (dim_u, dim_v, dim_w)
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_school = np.zeros((dim_u, rank_schoolbook))
    V_school = np.zeros((dim_v, rank_schoolbook))
    W_school = np.zeros((dim_w, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_school[u_idx, col] = 1.0
                V_school[v_idx, col] = 1.0
                W_school[w_idx, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank
    # We'll try ranks from 47 down to 40
    # If time runs out, we return the best found (which is at least schoolbook)
    
    for r in range(rank_schoolbook - 1, 39, -1):
        if time.time() > deadline - 20:
            break
            
        # Initialize U, V, W
        # Use random initialization
        np.random.seed(42 + r) # Reproducibility? Or just random. Let's use random for diversity.
        # Actually, deterministic is better for "improvement" if it works.
        # Let's try a few random inits for each rank.
        found = False
        for init_idx in range(5):
            if time.time() > deadline - 10:
                break
                
            U = np.random.randn(dim_u, r)
            V = np.random.randn(dim_v, r)
            W = np.random.randn(dim_w, r)
            
            # ALS Optimization
            # Minimize || T - sum_c U_c V_c W_c ||_F^2
            # This is equivalent to minimizing the tensor nuclear norm? No, just Frobenius norm of the difference.
            
            # Reshape T for easier matmul operations
            # T_uvw
            # We can update U by fixing V, W.
            # T_u.. = sum_c U_uc V_vc W_wc
            # For a fixed u, T_u.. = U_u. (V W^T)? No.
            # Let's use the standard ALS update for tensor decomposition.
            # T_{uvw} = \sum_c U_{uc} V_{vc} W_{wc}
            
            # We can flatten V and W into a matrix of shape (r, dim_v * dim_w)?
            # Or use einsum.
            
            # Update U:
            # U_{uc} = \sum_{v,w} T_{uvw} V_{vc} W_{wc} / \sum_{v,w} V_{vc}^2 W_{wc}^2 ?
            # This is the least squares solution for U_c given V, W.
            
            # Let's implement one ALS iteration.
            
            for it in range(20): # 20 iterations might be enough for convergence to a local min
                if time.time() > deadline - 10:
                    break
                    
                # Update U
                # T_reshaped: (dim_u, dim_v * dim_w)
                # V_W_reshaped: (dim_v * dim_w, r) where (V_W_reshaped)[v*w_idx, c] = V[v, c] * W[w, c]
                # This is outer product of V and W along c?
                # V is (dim_v, r), W is (dim_w, r).
                # We want a matrix M of shape (dim_v * dim_w, r) such that M[v, w, c] = V[v, c] * W[w, c].
                # This is np.einsum('vc,wc->vwc', V, W).reshape(dim_v*dim_w, r)
                
                VW = np.einsum('vc,wc->vwc', V, W).reshape(dim_v * dim_w, r)
                T_flat = T.reshape(dim_u, dim_v * dim_w)
                
                # U = T_flat @ VW @ (VW^T VW)^-1 ?
                # Least squares: min_U || T_flat - U VW^T ||_F^2
                # Solution: U = T_flat @ VW @ pinv(VW^T @ VW)
                
                # To save time, we can just do a few gradient steps or use np.linalg.lstsq.
                # np.linalg.lstsq is robust.
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VW.T, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                    
                # Update V
                UW = np.einsum('uc,wc->uwc', U, W).reshape(dim_u * dim_w, r)
                T_flat_v = T.reshape(dim_v, dim_u * dim_w).T # (dim_u * dim_w, dim_v) -> T is (u, v, w)
                # T is (u, v, w). We want to solve for V.
                # T_{uvw} = V_{vc} (U W)_{uc wc}
                # Reshape T to (dim_v, dim_u * dim_w)
                T_v = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(UW.T, T_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                    
                # Update W
                UV = np.einsum('uc,vc->uvc', U, V).reshape(dim_u * dim_v, r)
                T_w = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UV.T, T_w.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                    
            # Check residual
            # Reconstruct T_rec
            T_rec = np.einsum('uc,vc,wc->uvw', U, V, W)
            residual = np.linalg.norm(T - T_rec)
            
            if residual < 1e-9:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness
                T_rec_snap = np.einsum('uc,vc,wc->uvw', U_snap, V_snap, W_snap)
                if np.allclose(T, T_rec_snap, atol=1e-9):
                    best_rank = r
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    found = True
                    break
        
        if found:
            # We found a lower rank, continue to see if we can go lower
            # But we should break if we hit a hard lower bound or time limit
            continue
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
