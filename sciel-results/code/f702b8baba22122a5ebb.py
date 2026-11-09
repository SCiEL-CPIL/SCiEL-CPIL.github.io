import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 5, 6
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T
    # T[a, b, c] = 1 if consistent, 0 otherwise
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Trivial solution
    trivial_rank = n * m * p
    U_trivial = np.zeros((dim_u, trivial_rank))
    V_trivial = np.zeros((dim_v, trivial_rank))
    W_trivial = np.zeros((dim_w, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, col] = 1.0
                V_trivial[b, col] = 1.0
                W_trivial[c, col] = 1.0
                col += 1
                
    best_U, best_V, best_W, best_rank = U_trivial, V_trivial, W_trivial, trivial_rank
    
    # Search for lower rank using ALS
    # We try ranks from 59 down to 30
    for test_rank in range(59, 29, -1):
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try a few random initializations
        for trial in range(10):
            if time.time() > deadline - 20:
                break
                
            # Random initialization
            U = np.random.randn(dim_u, test_rank)
            V = np.random.randn(dim_v, test_rank)
            W = np.random.randn(dim_w, test_rank)
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline - 20:
                    break
                    
                # Update U
                # For each a, solve U[a, :] @ (V^T khatri W^T) = T[a, :, :]
                # M_vw[:, r] = V[:, r] * W[:, r] (element-wise product, flattened)
                # Actually, the contraction is sum_r U[a,r] V[b,r] W[c,r]
                # So for fixed V, W, we want U[a, :] to minimize
                # sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # This is equivalent to U[a, :] M = T[a, :, :] where M[:, r] = V[:, r] * W[:, r] (flattened)
                
                M_vw = np.zeros((dim_v * dim_w, test_rank))
                for r in range(test_rank):
                    M_vw[:, r] = (V[:, r] * W[:, r]).flatten()
                    
                # T_flat[a] is T[a, :, :].flatten()
                T_flat = T.flatten() # shape (dim_u * dim_v * dim_w)
                
                for a in range(dim_u):
                    y = T[a, :, :].flatten()
                    # Solve U[a, :] M_vw = y
                    # Use lstsq
                    sol, residuals, rank, sv = np.linalg.lstsq(M_vw, y, rcond=None)
                    U[a, :] = sol
                    
                # Update V
                M_uw = np.zeros((dim_u * dim_w, test_rank))
                for r in range(test_rank):
                    M_uw[:, r] = (U[:, r] * W[:, r]).flatten()
                    
                for b in range(dim_v):
                    # y is T[:, b, :].flatten()
                    # Need to reshape T to extract slice
                    y = T[:, b, :].flatten()
                    sol, residuals, rank, sv = np.linalg.lstsq(M_uw, y, rcond=None)
                    V[b, :] = sol
                    
                # Update W
                M_uv = np.zeros((dim_u * dim_v, test_rank))
                for r in range(test_rank):
                    M_uv[:, r] = (U[:, r] * V[:, r]).flatten()
                    
                for c in range(dim_w):
                    y = T[:, :, c].flatten()
                    sol, residuals, rank, sv = np.linalg.lstsq(M_uv, y, rcond=None)
                    W[c, :] = sol
                    
            # Check error
            # Compute tensor from U, V, W
            # T_est[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # We can compute this efficiently using einsum or loops
            # T_est = np.einsum('ar,br,cr->abc', U, V, W)
            T_est = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T - T_est)
            
            if error < 1e-5:
                best_U, best_V, best_W, best_rank = U, V, W, test_rank
                found = True
                break
        
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-START
# Entry point is construct(), returns dict with 'U', 'V', 'W' keys
# EVOLVE-BLOCK-END
