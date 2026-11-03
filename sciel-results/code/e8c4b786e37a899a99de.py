import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # 1. Construct the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Flatten T for vector operations
    # We will use slices, but having a flat version is good for final check
    T_flat = T.flatten()
    
    # 2. Schoolbook solution (Rank n*m*p)
    rank_sb = n * m * p
    U_sb = np.zeros((dim_u, rank_sb))
    V_sb = np.zeros((dim_v, rank_sb))
    W_sb = np.zeros((dim_w, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_sb[i * m + j, c] = 1
                V_sb[j * p + k, c] = 1
                W_sb[k * n + i, c] = 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # 3. Search for lower rank
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try ranks from 25 to 59
    for R in range(25, 60):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Use small random integers to help convergence to integer/half-integer solutions?
            # Or just random floats.
            U_curr = np.random.randn(dim_u, R)
            V_curr = np.random.randn(dim_v, R)
            W_curr = np.random.randn(dim_w, R)
            
            # Iterative Least Squares
            for iter in range(20):
                if time.time() > deadline:
                    break
                    
                # 1. Solve for U
                # M_U_block: (dim_v * dim_w) x R
                M_U_block = np.zeros((dim_v * dim_w, R))
                for r in range(R):
                    M_U_block[:, r] = (V_curr[:, r] * W_curr[:, r].T).flatten()
                
                U_sol = np.zeros((dim_u, R))
                for i in range(dim_u):
                    t_slice = T[i, :, :].flatten()
                    # Solve M_U_block * u_i = t_slice
                    u_i, _, _, _ = np.linalg.lstsq(M_U_block, t_slice, rcond=None)
                    U_sol[i, :] = u_i
                U_curr = U_sol
                
                # 2. Solve for V
                # M_V_block: (dim_u * dim_w) x R
                M_V_block = np.zeros((dim_u * dim_w, R))
                for r in range(R):
                    M_V_block[:, r] = (U_curr[:, r] * W_curr[:, r].T).flatten()
                
                V_sol = np.zeros((dim_v, R))
                for j in range(dim_v):
                    # t_slice: T[:, j, :] flattened
                    # T shape (dim_u, dim_v, dim_w)
                    # Slice j along axis 1
                    t_slice = T[:, j, :].flatten()
                    v_j, _, _, _ = np.linalg.lstsq(M_V_block, t_slice, rcond=None)
                    V_sol[j, :] = v_j
                V_curr = V_sol
                
                # 3. Solve for W
                # M_W_block: (dim_u * dim_v) x R
                M_W_block = np.zeros((dim_u * dim_v, R))
                for r in range(R):
                    M_W_block[:, r] = (U_curr[:, r] * V_curr[:, r].T).flatten()
                
                W_sol = np.zeros((dim_w, R))
                for k in range(dim_w):
                    # t_slice: T[:, :, k] flattened
                    t_slice = T[:, :, k].flatten()
                    w_k, _, _, _ = np.linalg.lstsq(M_W_block, t_slice, rcond=None)
                    W_sol[k, :] = w_k
                W_curr = W_sol
                
            # Check the error
            T_recon = np.zeros((dim_u, dim_v, dim_w), dtype=float)
            for r in range(R):
                T_recon += np.einsum('i,j,k->ijk', U_curr[:, r], V_curr[:, r], W_curr[:, r])
                
            error = np.linalg.norm(T_recon.flatten() - T_flat)
            
            if error < 1e-6:
                # Found a solution!
                # Round to half-integers
                U_int = np.round(U_curr * 2) / 2.0
                V_int = np.round(V_curr * 2) / 2.0
                W_int = np.round(W_curr * 2) / 2.0
                
                # Verify
                T_recon_int = np.zeros((dim_u, dim_v, dim_w), dtype=float)
                for r in range(R):
                    T_recon_int += np.einsum('i,j,k->ijk', U_int[:, r], V_int[:, r], W_int[:, r])
                
                error_int = np.linalg.norm(T_recon_int.flatten() - T_flat)
                
                if error_int < 1e-6:
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    best_rank = R
                    # Return immediately as we want the lowest rank
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                    
    # If no better solution found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
