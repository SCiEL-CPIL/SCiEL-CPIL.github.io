import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 2, 11
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Build the rank-39 decomposition as baseline
    total_rank = 39
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h)
        U[0, col] = 1; U[3, col] = 1
        V[v_e, col] = 1; V[v_h, col] = 1
        W[w_c00, col] = 1; W[w_c11, col] = 1
        col += 1
        
        # m2 = (c+d)e
        U[2, col] = 1; U[3, col] = 1
        V[v_e, col] = 1
        W[w_c10, col] = 1; W[w_c11, col] = -1
        col += 1
        
        # m3 = a(f-h)
        U[0, col] = 1
        V[v_f, col] = 1; V[v_h, col] = -1
        W[w_c01, col] = 1; W[w_c11, col] = 1
        col += 1
        
        # m4 = d(g-e)
        U[3, col] = 1
        V[v_g, col] = 1; V[v_e, col] = -1
        W[w_c00, col] = 1; W[w_c10, col] = 1
        col += 1
        
        # m5 = (a+b)h
        U[0, col] = 1; U[1, col] = 1
        V[v_h, col] = 1
        W[w_c00, col] = -1; W[w_c01, col] = 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[2, col] = 1; U[0, col] = -1
        V[v_e, col] = 1; V[v_f, col] = 1
        W[w_c11, col] = 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[1, col] = 1; U[3, col] = -1
        V[v_g, col] = 1; V[v_h, col] = 1
        W[w_c00, col] = 1
        col += 1
    
    # Single column k2 = 10
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    U[0, col] = 1; V[v_e2, col] = 1; W[w_c02, col] = 1; col += 1
    U[1, col] = 1; V[v_g2, col] = 1; W[w_c02, col] = 1; col += 1
    U[2, col] = 1; V[v_e2, col] = 1; W[w_c12, col] = 1; col += 1
    U[3, col] = 1; V[v_g2, col] = 1; W[w_c12, col] = 1; col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = total_rank
    
    # Try to find a lower rank decomposition using a different strategy
    # Strategy: Try rank 38 by attempting to merge two schoolbook terms into a Strassen-like term
    
    # Alternative: Try a completely different decomposition structure
    # For 2x2 by 2x11, we can view it as 11 independent 2x1 multiplications
    # Each 2x2 by 2x1 has tensor of size (4, 2, 2)
    # The minimal rank for 2x2 by 2x1 is actually 4 (schoolbook) since it's a rank-1 slice
    
    # Let's try a numerical approach: use SVD-based rank minimization
    # Build the full tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a low-rank decomposition by iterative projection
    # Start with rank 38 and try to fit
    
    def try_rank(r, max_iter=50):
        """Try to find a rank-r decomposition using iterative method."""
        if time.time() > deadline - 20:
            return None
        
        # Initialize with random or structured
        U_init = np.zeros((n*m, r))
        V_init = np.zeros((m*p, r))
        W_init = np.zeros((p*n, r))
        
        # Use the existing decomposition as initialization for first min(r, 39) columns
        init_cols = min(r, 39)
        if init_cols > 0:
            U_init[:, :init_cols] = U[:, :init_cols]
            V_init[:, :init_cols] = V[:, :init_cols]
            W_init[:, :init_cols] = W[:, :init_cols]
        
        # For remaining columns, use random small values
        if r > init_cols:
            U_init[:, init_cols:] = np.random.randn(n*m, r - init_cols) * 0.1
            V_init[:, init_cols:] = np.random.randn(m*p, r - init_cols) * 0.1
            W_init[:, init_cols:] = np.random.randn(p*n, r - init_cols) * 0.1
        
        # Iterative refinement: alternate between fixing two factors and solving for the third
        for iteration in range(max_iter):
            if time.time() > deadline - 20:
                return None
            
            # Fix V and W, solve for U
            # T = sum_r U[:,r] outer V[:,r] outer W[:,r]
            # For each (i,j) in U: T[i,j,:] = sum_r U[i,j,r] * (V[:,r] outer W[:,r]).reshape(-1)
            # This is a linear system: T_slice = U_row @ M where M[r, (k,l)] = V[k,r]*W[l,r]
            
            # Build the matrix M for U
            # M has shape (r, m*p * p*n)
            M_U = np.zeros((r, m*p * p*n))
            for r_idx in range(r):
                M_U[r_idx, :] = (V_init[:, r_idx] * W_init[:, r_idx]).reshape(-1)
            
            # T reshaped: for each (i,j), we have a vector of length m*p*p*n
            # T[i*m+j, :, :] reshaped to (m*p*p*n,)
            T_flat = T.reshape(n*m, m*p*p*n)
            
            # Solve: T_flat[i*m+j, :] = U[i*m+j, :] @ M_U
            # U[i*m+j, :] = T_flat[i*m+j, :] @ M_U^+ (pseudo-inverse)
            try:
                U_new = T_flat @ np.linalg.pinv(M_U)
            except:
                U_new = U_init
            
            # Fix U and W, solve for V
            M_V = np.zeros((r, n*m * p*n))
            for r_idx in range(r):
                M_V[r_idx, :] = (U_init[:, r_idx] * W_init[:, r_idx]).reshape(-1)
            
            T_flat_V = T.reshape(m*p, n*m*p*n)
            try:
                V_new = T_flat_V @ np.linalg.pinv(M_V)
            except:
                V_new = V_init
            
            # Fix U and V, solve for W
            M_W = np.zeros((r, n*m * m*p))
            for r_idx in range(r):
                M_W[r_idx, :] = (U_init[:, r_idx] * V_init[:, r_idx]).reshape(-1)
            
            T_flat_W = T.reshape(p*n, n*m*m*p)
            try:
                W_new = T_flat_W @ np.linalg.pinv(M_W)
            except:
                W_new = W_init
            
            U_init = U_new
            V_init = V_new
            W_init = W_new
        
        # Check if the decomposition is correct
        T_recon = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            T_recon += np.einsum('a,b,c->abc', U_init[:, r_idx], V_init[:, r_idx], W_init[:, r_idx])
        
        if np.allclose(T_recon, T, atol=1e-8):
            return U_init, V_init, W_init
        
        return None
    
    # Try ranks from 38 down to 35
    for target_rank in range(38, 34, -1):
        if time.time() > deadline - 30:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = target_rank
            break
    
    # Verify the best solution
    T_recon = np.zeros((n*m, m*p, p*n))
    for r_idx in range(best_rank):
        T_recon += np.einsum('a,b,c->abc', best_U[:, r_idx], best_V[:, r_idx], best_W[:, r_idx])
    
    if not np.allclose(T_recon, T, atol=1e-8):
        # Fallback to the known rank-39 solution
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = 39
    
    # Round to nearest half-integer if possible
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    best_U_rounded = round_to_half(best_U)
    best_V_rounded = round_to_half(best_V)
    best_W_rounded = round_to_half(best_W)
    
    # Check if rounded version still works
    T_recon_rounded = np.zeros((n*m, m*p, p*n))
    for r_idx in range(best_rank):
        T_recon_rounded += np.einsum('a,b,c->abc', best_U_rounded[:, r_idx], best_V_rounded[:, r_idx], best_W_rounded[:, r_idx])
    
    if np.allclose(T_recon_rounded, T, atol=1e-8):
        best_U, best_V, best_W = best_U_rounded, best_V_rounded, best_W_rounded
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

