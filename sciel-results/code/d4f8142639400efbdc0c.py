import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Precompute the target tensor in flattened form
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # We want sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find lower rank decompositions using ALS
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from rank_sb-1 down, with time-limited search
    for R in range(rank_sb - 1, max(rank_sb - 30, 1), -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # ALS: alternate solving linear systems
            # For fixed V, W: solve for U (linear in each row)
            # T[a,:,:] = sum_r U[a,r] * (V[:,:,r] * W[:,:,r])
            # So U[a,r] solves: sum_r U[a,r] * B_r = T[a,:,:]
            # where B_r = V[:,:,r] * W[:,:,r] (outer product flattened)
            
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Update U: for each row a, solve linear system
                # T[a] = sum_r U[a,r] * (V[:,r] * W[:,r])
                # Let M_r = V[:,r] * W[:,r] (element-wise product, shape m*p x p*n)
                # Flatten M_r to vector of length (m*p)*(p*n)
                # Then T[a].flatten() = sum_r U[a,r] * M_r.flatten()
                # This is a linear system: U[a,:] @ M_flat = T[a].flatten()
                
                # Build matrix A where A[:, r] = M_r.flatten()
                # A has shape (m*p*p*n, R)
                A = np.zeros((m * p * p * n, R))
                for r in range(R):
                    M_r = V[:, r] * W[:, r]  # shape (m*p, p*n)
                    A[:, r] = M_r.flatten()
                
                # For each a, solve: U[a,:] @ A = T[a].flatten()
                # This is: A.T @ U[a,:].T = T[a].flatten()
                # Use least squares
                U_new = np.zeros((n * m, R))
                for a in range(n * m):
                    # Solve A.T @ x = T[a].flatten()
                    # x = U[a,:]
                    x, _, _, _ = np.linalg.lstsq(A.T, T[a].flatten(), rcond=None)
                    U_new[a, :] = x
                U = U_new
                
                # Update V similarly
                # T[:,b,:] = sum_r V[b,r] * (U[:,r] * W[:,r])
                # Let N_r = U[:,r] * W[:,r] (shape n*m x p*n)
                # T[:,b,:].flatten() = sum_r V[b,r] * N_r.flatten()
                B_mat = np.zeros((n * m * p * n, R))
                for r in range(R):
                    N_r = U[:, r] * W[:, r]  # shape (n*m, p*n)
                    B_mat[:, r] = N_r.flatten()
                
                V_new = np.zeros((m * p, R))
                for b in range(m * p):
                    # We need T[:,b,:].flatten()
                    # T has shape (n*m, m*p, p*n), so T[:,b,:] has shape (n*m, p*n)
                    x, _, _, _ = np.linalg.lstsq(B_mat.T, T[:, b, :].flatten(), rcond=None)
                    V_new[b, :] = x
                V = V_new
                
                # Update W
                # T[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r])
                # Let P_r = U[:,r] * V[:,r] (shape n*m x m*p)
                # T[:,:,c].flatten() = sum_r W[c,r] * P_r.flatten()
                C_mat = np.zeros((n * m * m * p, R))
                for r in range(R):
                    P_r = U[:, r] * V[:, r]  # shape (n*m, m*p)
                    C_mat[:, r] = P_r.flatten()
                
                W_new = np.zeros((p * n, R))
                for c in range(p * n):
                    # T[:,:,c].flatten() has shape (n*m*m*p,)
                    x, _, _, _ = np.linalg.lstsq(C_mat.T, T[:, :, c].flatten(), rcond=None)
                    W_new[c, :] = x
                W = W_new
                
                # Check convergence
                if iteration > 0:
                    # Compute residual
                    # residual = sum_r U[:,r] * V[:,r] * W[:,r] - T
                    # This is expensive, so check every few iterations
                    if iteration % 10 == 0:
                        residual = 0.0
                        for r in range(R):
                            residual += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
                        residual -= T
                        err = np.linalg.norm(residual)
                        if err < 1e-8:
                            found = True
                            break
                
                if found:
                    break
            
            if found:
                best_rank = R
                best_U, best_V, best_W = U, V, W
                break  # Move to next lower rank
        
        if found:
            # Try one more rank lower
            continue
        else:
            # If we couldn't find rank R, try R+1? No, just break and return best
            # Actually, let's try a few more ranks
            if R > best_rank - 5:
                continue
            else:
                break
    
    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
