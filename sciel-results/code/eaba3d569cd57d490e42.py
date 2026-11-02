import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions of the factor matrices
    dim_u = n * m  # 16
    dim_v = m * p  # 36
    dim_w = p * n  # 36
    
    # Construct the target tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    best_result = None
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Try ranks from 143 down to 100
    # If we find a valid one, we can stop or try to go lower
    # Given the time, we might not go very low, but 143 is a start.
    # Actually, let's try a few random restarts for a specific rank first.
    # If we can't beat 144, we return 144.
    
    # Let's try rank 140, 130, 120...
    # Or just do a search.
    
    for R in range(143, 99, -5):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(42 + restart)
            U = np.random.randn(dim_u, R) * 0.1
            V = np.random.randn(dim_v, R) * 0.1
            W = np.random.randn(dim_w, R) * 0.1
            
            for iter in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U
                # For each u, solve U[u, :] M^T = T[u, :, :]
                # M[v, w, c] = V[v, c] W[w, c]
                # We can compute M for all c at once?
                # M is dim_v x dim_w x R
                # M = V[:, :, None] * W[None, :, :] -> shape (dim_v, dim_w, R)
                # We need to solve for U[u, :] such that sum_c U[u, c] M[:, :, c] = T[u, :, :]
                # This is equivalent to:
                # Let M_flat be dim_v*dim_w x R.
                # M_flat[(v, w), c] = M[v, w, c]
                # T_flat[u] is dim_v*dim_w vector.
                # U[u, :] M_flat^T = T_flat[u]
                
                M = V[:, :, None] * W[None, :, :] # (dim_v, dim_w, R)
                M_flat = M.transpose(2, 0, 1).reshape(R, dim_v * dim_w).T # (dim_v*dim_w, R)
                
                for u in range(dim_u):
                    t_u = T[u, :, :].flatten()
                    # Solve U[u, :] M_flat^T = t_u
                    # This is (R x 1) * (R x dim_v*dim_w)^T ?
                    # We want x (R) such that x @ M_flat.T = t_u
                    # M_flat.T is (R, dim_v*dim_w)
                    # x @ M_flat.T = t_u -> x^T = (M_flat.T)^T x? No.
                    # x @ M_flat.T = t_u
                    # M_flat.T @ x^T = t_u^T
                    # We can use lstsq: x = lstsq(M_flat.T, t_u)
                    # Wait, M_flat.T is R x (V*W). t_u is V*W.
                    # We want x (R) such that M_flat.T @ x = t_u?
                    # Let's check dimensions.
                    # M_flat is (V*W, R).
                    # M_flat.T is (R, V*W).
                    # x is (R,).
                    # M_flat.T @ x is (V*W,).
                    # So we solve M_flat.T @ x = t_u.
                    x, _, _, _ = np.linalg.lstsq(M_flat.T, t_u, rcond=None)
                    U[u, :] = x
                
                # Update V
                # For each v, solve V[v, :] N^T = T[:, v, :]
                # N[u, w, c] = U[u, c] W[w, c]
                N = U[:, :, None] * W[None, :, :] # (dim_u, dim_w, R)
                N_flat = N.transpose(2, 0, 1).reshape(R, dim_u * dim_w).T # (dim_u*dim_w, R)
                
                for v in range(dim_v):
                    t_v = T[:, v, :].flatten()
                    x, _, _, _ = np.linalg.lstsq(N_flat.T, t_v, rcond=None)
                    V[v, :] = x
                    
                # Update W
                # For each w, solve W[w, :] P^T = T[:, :, w]
                P = U[:, :, None] * V[None, :, :] # (dim_u, dim_v, R)
                P_flat = P.transpose(2, 0, 1).reshape(R, dim_u * dim_v).T # (dim_u*dim_v, R)
                
                for w in range(dim_w):
                    t_w = T[:, :, w].flatten()
                    x, _, _, _ = np.linalg.lstsq(P_flat.T, t_w, rcond=None)
                    W[w, :] = x
                    
                # Check error
                # T_recon = sum_c U[:, c] V[:, c] W[:, c]
                # Use einsum
                T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
                error = np.linalg.norm(T_recon - T)
                if error < 1e-5:
                    # Try to round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    T_recon_r = np.einsum('uc,vc,wc->uvw', U_rounded, V_rounded, W_rounded)
                    error_r = np.linalg.norm(T_recon_r - T)
                    if error_r < 1e-9:
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    else:
                        # If rounding fails, maybe return the float version if allowed?
                        # Prompt says "integer/half-integer entries".
                        # So we must round. If it doesn't work, this rank failed.
                        pass

    # If no better rank found, return schoolbook
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
