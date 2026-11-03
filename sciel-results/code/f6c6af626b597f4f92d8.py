import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor slices
    # T_w is a matrix of size (n*m) x (m*p)
    # T_w[u, v] = 1 if u, v, w correspond to a valid multiplication
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    T_slices = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_slices[w, u, v] = 1
    
    # Initialize with schoolbook as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    # Strategy: Use randomized SVD on flattened tensor to guide rank reduction
    # Flatten T into a 2D matrix: (p*n) x (n*m*m*p)
    # But this is too large. Instead, use random projections.
    
    # Try ranks from known lower bounds upward
    # Known: rank >= max(n*p, m*p, n*m) = max(18, 30, 15) = 30
    # Also rank >= (n*p + m*p + n*m - 1) / something... 
    # Let's try from 30 to rank_schoolbook
    
    for R in range(30, rank_schoolbook + 1):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        # Try multiple random initializations guided by SVD
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            
            # Use randomized SVD to get initial U, V
            # Create random projection matrix
            Omega = np.random.randn(n * m * m * p, max(R, 20))
            
            # Project T onto random subspace
            # T_flatten has shape (p*n, n*m*m*p)
            T_flatten = T_slices.reshape(p * n, -1)
            Y = T_flatten @ Omega  # (p*n, max(R,20))
            
            # SVD of Y to get basis
            try:
                U_y, S, Vt = np.linalg.svd(Y, full_matrices=False)
                # Take top R components
                U_basis = U_y[:, :R]  # (p*n, R)
                V_basis = Vt[:R, :]   # (R, max(R,20))
                # Get basis for the second mode
                Q = Omega @ V_basis  # (n*m*m*p, R)
            except:
                continue
            
            # Reshape Q to (n*m, m*p, R)
            Q_reshaped = Q.reshape(n * m, m * p, R)
            
            # For each r, Q_reshaped[:,:,r] should be approximately U_r * V_r^T
            # Use SVD on each slice to extract U_r, V_r
            U_curr = np.zeros((n * m, R))
            V_curr = np.zeros((m * p, R))
            
            for r in range(R):
                slice_r = Q_reshaped[:, :, r]
                try:
                    Ur, Sr, Vr = np.linalg.svd(slice_r, full_matrices=False)
                    # Take rank-1 approximation
                    U_curr[:, r] = Ur[:, 0] * Sr[0]
                    V_curr[:, r] = Vr[0]
                except:
                    U_curr[:, r] = np.random.randn(n * m)
                    V_curr[:, r] = np.random.randn(m * p)
            
            # Round to half-integers
            U_rounded = np.round(U_curr * 2) / 2
            V_rounded = np.round(V_curr * 2) / 2
            
            # Solve for W given U_rounded, V_rounded
            A = np.zeros((n * m * m * p, R))
            for r in range(R):
                M_r = np.outer(U_rounded[:, r], V_rounded[:, r])
                A[:, r] = M_r.flatten()
            
            W_solution = np.zeros((p * n, R))
            residual_sum = 0
            for w in range(p * n):
                t_vec = T_slices[w].flatten()
                W_w, residuals, rank_A, s = np.linalg.lstsq(A, t_vec, rcond=None)
                W_solution[w] = W_w
                residual_sum += np.linalg.norm(A @ W_w - t_vec)
            
            # Check if residual is small enough for rounding to work
            if residual_sum < 1e-3:
                W_rounded = np.round(W_solution * 2) / 2
                
                # Verify exactness
                T_recon = np.zeros((p * n, n * m, m * p))
                for w in range(p * n):
                    for r in range(R):
                        if W_rounded[w, r] != 0:
                            T_recon[w] += W_rounded[w, r] * np.outer(U_rounded[:, r], V_rounded[:, r])
                
                diff = np.abs(T_recon - T_slices)
                if np.max(diff) < 1e-9:
                    best_rank = R
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
