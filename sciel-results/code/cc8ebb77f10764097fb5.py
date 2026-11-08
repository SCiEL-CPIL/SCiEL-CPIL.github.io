import numpy as np
import time

def _find_block_decomposition(n, m, p, target_rank, time_limit=5.0):
    """
    Attempts to find a rank-minimal decomposition for the block multiplication.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    np.random.seed(42)
    
    # Target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1.0

    T_A = T.reshape(n*m, m*p * p*n)
    T_B = T.reshape(m*p, n*m * p*n)
    T_C = T.reshape(p*n, n*m * m*p)

    for seed in range(50):
        if time.time() - start_time > time_limit:
            break
        np.random.seed(seed)
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        for _ in range(200): # Iterations per seed
            # Update U
            for r in range(target_rank):
                vec_vw = (V[:, r] * W[:, r]).reshape(m*p, p*n).flatten()
                numer = T_A @ vec_vw
                denom = np.sum(vec_vw**2)
                if denom > 1e-12:
                    U[:, r] = numer / denom
                else:
                    U[:, r] = 0
                    
            # Update V
            for r in range(target_rank):
                vec_uw = (U[:, r] * W[:, r]).reshape(n*m, p*n).flatten()
                numer = T_B @ vec_uw
                denom = np.sum(vec_uw**2)
                if denom > 1e-12:
                    V[:, r] = numer / denom
                else:
                    V[:, r] = 0
                    
            # Update W
            for r in range(target_rank):
                vec_uv = (U[:, r] * V[:, r]).reshape(n*m, m*p).flatten()
                numer = T_C @ vec_uv
                denom = np.sum(vec_uv**2)
                if denom > 1e-12:
                    W[:, r] = numer / denom
                else:
                    W[:, r] = 0
            
            # Check convergence / rounding
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            D = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            if np.allclose(D, T, atol=1e-9):
                return U_rounded, V_rounded, W_rounded
                
    return None
