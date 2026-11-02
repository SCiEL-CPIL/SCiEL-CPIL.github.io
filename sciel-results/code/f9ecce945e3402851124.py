import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    best_error = np.inf
    
    # Schoolbook solution as fallback
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    # Try ranks from 40 to 53
    for R in range(40, 54):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this rank
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Initialize V and W randomly
            np.random.seed(42 + restart) # Reproducibility per restart
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Iterate ALS
            converged = False
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                # T_reshaped (n*m, m*p*p*n) = U (n*m, R) @ K (R, m*p*p*n)
                # K[r, v*p*n + w] = V[v, r] * W[w, r]
                # Construct K
                # V is (m*p, R), W is (p*n, R)
                # K_3d = V[:, :, None] * W[None, :, :] -> (m*p, p*n, R)
                # K = K_3d.transpose(2, 0, 1).reshape(R, m*p*p*n)
                K = (V[:, :, None] * W[None, :, :]).transpose(2, 0, 1).reshape(R, -1)
                
                T_flat = T.reshape(n*m, -1)
                
                # Solve U K = T_flat
                # U = T_flat @ pinv(K)
                try:
                    U, residuals, rank_u, s = np.linalg.lstsq(K.T, T_flat.T, rcond=None)
                    U = U.T
                except:
                    break
                    
                # Solve for V
                # T_flat_v (m*p, n*m*p*n) = V (m*p, R) @ L (R, n*m*p*n)
                # L[r, u*p*n + w] = U[u, r] * W[w, r]
                L = (U[:, :, None] * W[None, :, :]).transpose(2, 0, 1).reshape(R, -1)
                T_flat_v = T.transpose(1, 0, 2).reshape(m*p, -1)
                
                try:
                    V, residuals, rank_v, s = np.linalg.lstsq(L.T, T_flat_v.T, rcond=None)
                    V = V.T
                except:
                    break
                    
                # Solve for W
                # T_flat_w (p*n, n*m*m*p) = W (p*n, R) @ M (R, n*m*m*p)
                # M[r, u*m*p + v] = U[u, r] * V[v, r]
                M = (U[:, :, None] * V[None, :, :]).transpose(2, 0, 1).reshape(R, -1)
                T_flat_w = T.transpose(2, 0, 1).reshape(p*n, -1)
                
                try:
                    W, residuals, rank_w, s = np.linalg.lstsq(M.T, T_flat_w.T, rcond=None)
                    W = W.T
                except:
                    break
                    
                # Check convergence
                T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
                err = np.linalg.norm(T_rec - T)
                
                if err < 1e-10:
                    converged = True
                    break
                
            if converged:
                # Try to round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_rec_rounded = np.einsum('ur,vr,wr->uvw', U_rounded, V_rounded, W_rounded)
                err_rounded = np.linalg.norm(T_rec_rounded - T)
                
                if err_rounded < 1e-10:
                    best_rank = R
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    break # Found a good solution for this rank, try next lower? No, R is increasing.
                    # We want the minimum rank. Since we iterate R from 40 upwards, 
                    # the first one found is the lowest rank.
                else:
                    # If rounded doesn't work, check if float is good enough
                    # The task asks for exact decomposition. 
                    # If float error is tiny, maybe it's the right solution but floating point noise.
                    # Let's try rounding to integers as well?
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    T_rec_int = np.einsum('ur,vr,wr->uvw', U_int, V_int, W_int)
                    err_int = np.linalg.norm(T_rec_int - T)
                    if err_int < 1e-10:
                        best_rank = R
                        best_U, best_V, best_W = U_int, V_int, W_int
                        break
                    else:
                        # Keep float solution as best candidate for this rank if better than current best
                        if err < best_error:
                            best_error = err
                            if best_rank > R:
                                best_rank = R
                                best_U, best_V, best_W = U, V, W
                # If we found a solution for rank R, we can break the restart loop
                # and try the next rank? No, we want the MINIMUM rank.
                # So if we found rank 40, we should stop and return 40.
                # If we found rank 41, we should stop and return 41?
                # The loop goes 40, 41, ...
                # So the first success is the minimum rank found.
                break
                
        if best_rank <= R:
            # We found a solution for rank R or lower.
            # Since we iterate R from 40 upwards, the first one found is the lowest.
            break
            
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
