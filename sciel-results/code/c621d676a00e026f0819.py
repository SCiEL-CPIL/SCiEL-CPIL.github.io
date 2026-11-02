import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Build the target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition (Rank 48)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Target ranks to try
    target_ranks = [45, 46, 47]
    
    # Pre-compute indices for error evaluation
    nz_idx = np.argwhere(T != 0)
    nz_vals = T[nz_idx[:, 0], nz_idx[:, 1], nz_idx[:, 2]]
    
    zero_mask = (T == 0)
    zero_idx = np.argwhere(zero_mask)
    np.random.seed(42)
    # Use a sample of zeros to keep the residual vector size manageable
    zero_sample_idx = zero_idx[np.random.choice(len(zero_idx), size=min(100, len(zero_idx)), replace=False)]
    
    all_idx = np.vstack([nz_idx, zero_sample_idx])
    all_vals = np.concatenate([nz_vals, np.zeros(len(zero_sample_idx))])
    
    # Sizes
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    def error_function(params, R):
        # Reshape
        U = params[:size_U * R].reshape(size_U, R)
        V = params[size_U * R : size_U * R + size_V * R].reshape(size_V, R)
        W = params[size_U * R + size_V * R :].reshape(size_W, R)
        
        # Select rows
        U_sel = U[all_idx[:, 0], :]
        V_sel = V[all_idx[:, 1], :]
        W_sel = W[all_idx[:, 2], :]
        
        # Compute reconstruction
        rec = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        # Error
        return rec - all_vals
    
    for R in target_ranks:
        if time.time() > deadline - 10:
            break
            
        # Try multiple initializations
        for trial in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + R * 100 + trial)
            
            # Random initialization
            U = np.random.randn(size_U, R) * 0.1
            V = np.random.randn(size_V, R) * 0.1
            W = np.random.randn(size_W, R) * 0.1
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = opt.least_squares(error_function, x0, args=(R,), method='trf', max_nfev=500)
            except:
                continue
                
            # Extract solution
            U_sol = result.x[:size_U * R].reshape(size_U, R)
            V_sol = result.x[size_U * R : size_U * R + size_V * R].reshape(size_V, R)
            W_sol = result.x[size_U * R + size_V * R :].reshape(size_W, R)
            
            # Check full tensor error
            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            full_error = np.linalg.norm(T - recon)
            
            if full_error < 1e-4:
                # Try rounding
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                recon_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                error_rounded = np.linalg.norm(T - recon_rounded)
                
                if error_rounded < 1e-6:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = R
                    break
        
        if best_rank < rank_school:
            if best_rank == 45:
                break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
