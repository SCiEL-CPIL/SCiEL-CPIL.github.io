import time
import numpy as np

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 120
    
    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb), dtype=float)
    V_fb = np.zeros((m*p, rank_fb), dtype=float)
    W_fb = np.zeros((p*n, rank_fb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fb
    
    def check_decomp(U, V, W):
        T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
        return np.allclose(T_decomp, T, atol=1e-9)
    
    def snap_to_half(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s
    
    # Try to find rank 32 decomposition using ALS
    def try_rank_als(rank, max_iters=100):
        size_u, size_v, size_w = n*m, m*p, p*n
        
        # Initialize
        np.random.seed(42)
        U = np.random.randn(size_u, rank) * 0.1
        V = np.random.randn(size_v, rank) * 0.1
        W = np.random.randn(size_w, rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Update U
            for a in range(size_u):
                M = np.einsum('bt,ct->bct', V, W).reshape(size_v * size_w, rank)
                b_vec = T[a, :, :].reshape(-1)
                try:
                    U[a, :] = np.linalg.lstsq(M.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Update V
            for b in range(size_v):
                M = np.einsum('at,ct->act', U, W).reshape(size_u * size_w, rank)
                b_vec = T[:, b, :].reshape(-1)
                try:
                    V[b, :] = np.linalg.lstsq(M.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Update W
            for c in range(size_w):
                M = np.einsum('at,bt->abt', U, V).reshape(size_u * size_v, rank)
                b_vec = T[:, :, c].reshape(-1)
                try:
                    W[c, :] = np.linalg.lstsq(M.T, b_vec, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
            err = np.sum((T - T_decomp) ** 2)
            if err < 1e-10:
                break
        
        # Try to snap to half-integers
        U_s, V_s, W_s = snap_to_half(U, V, W)
        if check_decomp(U_s, V_s, W_s):
            return U_s, V_s, W_s
        elif check_decomp(U, V, W):
            # Check if values are close to half-integers
            if np.allclose(U, U_s, atol=1e-6) and np.allclose(V, V_s, atol=1e-6) and np.allclose(W, W_s, atol=1e-6):
                return U_s, V_s, W_s
        return None
    
    # Try rank 32
    result = try_rank_als(32, max_iters=50)
    if result is not None:
        U, V, W = result
        best_U, best_V, best_W = U, V, W
        best_rank = 32
    
    # If we found rank 32, return it
    if best_rank == 32:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
