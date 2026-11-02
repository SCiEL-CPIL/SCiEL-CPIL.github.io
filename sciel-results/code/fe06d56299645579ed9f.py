import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Build the multiplication tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if this term contributes to C[i, k] = sum_j A[i, j] B[j, k]
    # Wait, the tensor definition in the task is:
    # U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # The decomposition is T = sum_r U_r V_r W_r
    # Where T is the multiplication tensor.
    # The indices for U are (i, j), for V are (j, k), for W are (k, i).
    # The tensor T has non-zero entries at ( (i,j), (j,k), (k,i) ) for all i,j,k.
    # Specifically, T[ (i,j), (j,k), (k,i) ] = 1.
    
    # Let's create the tensor as a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # We want to find U (n*m, R), V (m*p, R), W (p*n, R)
    # such that T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
    
    # We will try to find a solution for ranks R from 40 to 53.
    # We use an alternating least squares (ALS) approach.
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Try ranks from 40 to 53
    # We prefer lower ranks, so we try 40, 41, ...
    # But higher ranks are easier to find.
    # Let's try 46 first, then 47, ..., 53.
    # If 46 fails, try 47, etc.
    
    # Actually, let's try a few random restarts for each rank.
    
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Schoolbook solution
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
                
    for R in range(40, 54):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize V and W randomly
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Iterate ALS
            converged = False
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
                # Reshape T to (n*m, m*p * p*n)
                T_mat = T.reshape(n*m, m*p * p*n)
                # M[v, w, r] = V[v, r] W[w, r]
                # We need a matrix M of shape (m*p * p*n, R)
                # M_idx = v * p*n + w
                # M[M_idx, r] = V[v, r] * W[w, r]
                M = np.zeros((m*p * p*n, R))
                for v in range(m*p):
                    for w in range(p*n):
                        M[v * p*n + w, :] = V[v, :] * W[w, :]
                
                # Solve U M^T = T_mat
                # U = T_mat (M^T)^+
                # Using least squares
                try:
                    U, residuals, rank_u, s = np.linalg.lstsq(M.T, T_mat.T, rcond=None)
                    U = U.T
                except:
                    break
                    
                # Check convergence
                # Reconstruct T
                T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
                err = np.linalg.norm(T_rec - T)
                
                if err < 1e-9:
                    converged = True
                    break
                
                # Solve for V
                # T[u, v, w] = sum_r V[v, r] (U[u, r] W[w, r])
                # Reshape T to (m*p, n*m * p*n)
                T_mat_v = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
                M_v = np.zeros((n*m * p*n, R))
                for u in range(n*m):
                    for w in range(p*n):
                        M_v[u * p*n + w, :] = U[u, :] * W[w, :]
                        
                try:
                    V, residuals, rank_v, s = np.linalg.lstsq(M_v.T, T_mat_v.T, rcond=None)
                    V = V.T
                except:
                    break
                    
                # Solve for W
                T_mat_w = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
                M_w = np.zeros((n*m * m*p, R))
                for u in range(n*m):
                    for v in range(m*p):
                        M_w[u * m*p + v, :] = U[u, :] * V[v, :]
                        
                try:
                    W, residuals, rank_w, s = np.linalg.lstsq(M_w.T, T_mat_w.T, rcond=None)
                    W = W.T
                except:
                    break
                    
            if converged:
                best_rank = R
                best_U, best_V, best_W = U, V, W
                break # Found a solution for this rank, try next lower rank? No, we are iterating 40..53.
                # So if we find 40, we break the restart loop, then the R loop.
                
        if best_rank < best_rank: # This logic is wrong, best_rank is updated inside.
            pass
            
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
