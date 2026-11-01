"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _fast_als_decompose(T, rank, max_iter=200, tol=1e-12, seed=42):
    """Fast ALS for tensor decomposition using matrix operations."""
    rng = np.random.RandomState(seed)
    dim1, dim2, dim3 = T.shape
    
    U = rng.randn(dim1, rank) * 0.5
    V = rng.randn(dim2, rank) * 0.5
    W = rng.randn(dim3, rank) * 0.5
    
    # Precompute matricizations
    T_u1 = T.reshape(dim1, dim2 * dim3)
    T_u2 = T.reshape(dim2, dim1 * dim3)
    T_u3 = T.reshape(dim3, dim1 * dim2)
    
    for it in range(max_iter):
        # Update U: for each column c, U[:,c] = (V[:,c] ⊗ W[:,c])^T @ T_u1 / ||V[:,c] ⊗ W[:,c]||^2
        # But we need to exclude component c from the residual
        # Efficient: U = (V ⊙ W)^T @ T_u1 / (V ⊙ W)^T @ (V ⊙ W) where ⊙ is Khatri-Rao
        # Khatri-Rao: columns are element-wise products
        V_krao_W = np.multiply.outer(V, W)  # (dim2*dim3, rank) - actually this is Kronecker, not Khatri-Rao
        # Khatri-Rao product of V (dim2×rank) and W (dim3×rank): result is (dim2*dim3)×rank
        # where column c is V[:,c] ⊗ W[:,c]
        V_krao_W = np.zeros((dim2 * dim3, rank))
        for c in range(rank):
            V_krao_W[:, c] = np.outer(V[:, c], W[:, c]).flatten()
        
        # Solve least squares for U
        A = V_krao_W  # (dim2*dim3, rank)
        # U^T = (A^T A)^{-1} A^T T_u1^T
        try:
            U = np.linalg.lstsq(A, T_u1.T, rcond=None)[0].T
        except:
            pass
        
        # Update V
        U_krao_W = np.zeros((dim1 * dim3, rank))
        for c in range(rank):
            U_krao_W[:, c] = np.outer(U[:, c], W[:, c]).flatten()
        try:
            V = np.linalg.lstsq(U_krao_W, T_u2.T, rcond=None)[0].T
        except:
            pass
        
        # Update W
        U_krao_V = np.zeros((dim1 * dim2, rank))
        for c in range(rank):
            U_krao_V[:, c] = np.outer(U[:, c], V[:, c]).flatten()
        try:
            W = np.linalg.lstsq(U_krao_V, T_u3.T, rcond=None)[0].T
        except:
            pass
        
        # Check convergence
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(T - recon) / np.linalg.norm(T)
        if err < tol:
            break
    
    return U, V, W, err

def _try_round_and_verify(U, V, W, T):
    """Try to round to integers/half-integers and verify exactness."""
    dim1, dim2, dim3 = T.shape
    rank = U.shape[1]
    
    for scale in [1, 2]:
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        
        # Verify
        recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
        if np.allclose(T, recon, atol=1e-9):
            return U_r, V_r, W_r, True
    
    # Try local search on integer grid
    best_err = float('inf')
    best = None
    for scale in [1, 2]:
        U_r = np.round(U * scale).astype(int) / scale
        V_r = np.round(V * scale).astype(int) / scale
        W_r = np.round(W * scale).astype(int) / scale
        
        for _ in range(10):
            recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
            err = np.linalg.norm(T - recon)
            if err < best_err:
                best_err = err
                best = (U_r.copy(), V_r.copy(), W_r.copy())
            if err < 1e-9:
                return U_r, V_r, W_r, True
            
            # Try flipping some entries
            improved = False
            for _ in range(30):
                mat = np.random.choice(['U', 'V', 'W'])
                idx = np.random.choice(U_r.size if mat == 'U' else V_r.size if mat == 'V' else W_r.size)
                i, c = np.unravel_index(idx, (dim1 if mat=='U' else dim2 if mat=='V' else dim3, rank))
                orig = (U_r if mat=='U' else V_r if mat=='V' else W_r)[i, c]
                for delta in [-1.0/scale, 1.0/scale]:
                    (U_r if mat=='U' else V_r if mat=='V' else W_r)[i, c] = orig + delta
                    recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                    err_new = np.linalg.norm(T - recon)
                    if err_new < err:
                        err = err_new
                        improved = True
                    else:
                        (U_r if mat=='U' else V_r if mat=='V' else W_r)[i, c] = orig
                if not improved:
                    break
            if not improved:
                break
    
    if best is not None and best_err < 1e-9:
        return best[0], best[1], best[2], True
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    T = _build_target_tensor(n, m, p)
    dim1, dim2, dim3 = T.shape
    
    # Schoolbook fallback
    sb_rank = n * m * p
    U_sb = [[0] * sb_rank for _ in range(n * m)]
    V_sb = [[0] * sb_rank for _ in range(m * p)]
    W_sb = [[0] * sb_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    # Try decreasing ranks
    for target_rank in [32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 25:
                break
            
            U, V, W, err = _fast_als_decompose(T, target_rank, max_iter=100, tol=1e-12, seed=seed)
            
            if err < 1e-6:
                U_r, V_r, W_r, ok = _try_round_and_verify(U, V, W, T)
                if ok:
                    U_list = [[float(U_r[i, c]) for c in range(target_rank)] for i in range(dim1)]
                    V_list = [[float(V_r[i, c]) for c in range(target_rank)] for i in range(dim2)]
                    W_list = [[float(W_r[i, c]) for c in range(target_rank)] for i in range(dim3)]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
