"""Matrix multiplication 3,3,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-10):
    T = _build_target_tensor(n, m, p)
    rank = U.shape[1]
    T_recon = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_recon, T, atol=tol, rtol=0)

def _als_decompose(T, r, max_iter=50, seed=42, tol=1e-12):
    """ALS for CP decomposition of tensor T."""
    rng = np.random.RandomState(seed)
    shape = T.shape
    U = rng.randn(shape[0], r)
    V = rng.randn(shape[1], r)
    W = rng.randn(shape[2], r)
    
    # Normalize
    for col in range(r):
        nu = np.linalg.norm(U[:, col])
        nv = np.linalg.norm(V[:, col])
        nw = np.linalg.norm(W[:, col])
        if nu > 1e-15 and nv > 1e-15 and nw > 1e-15:
            scale = (nu * nv * nw) ** (1.0/3.0)
            U[:, col] /= scale
            V[:, col] /= scale
            W[:, col] /= scale
    
    for it in range(max_iter):
        # Update U: T_1 @ (V ⊙ W)^+
        # T_1: shape (shape[0], shape[1]*shape[2])
        T1 = T.transpose(0, 1, 2).reshape(shape[0], -1)
        VW = np.kron(V.T, W.T).T  # Khatri-Rao: each column is v_r ⊗ w_r
        # Actually, Khatri-Rao product: column r is V[:,r] ⊗ W[:,r]
        VW = np.zeros((shape[1] * shape[2], r))
        for r_idx in range(r):
            VW[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx])
        try:
            U_new, _, _, _ = np.linalg.lstsq(VW, T1, rcond=None)
            U = U_new
        except:
            pass
        
        # Update V: T_2 @ (U ⊙ W)^+
        T2 = T.transpose(1, 0, 2).reshape(shape[1], -1)
        UW = np.zeros((shape[0] * shape[2], r))
        for r_idx in range(r):
            UW[:, r_idx] = np.kron(U[:, r_idx], W[:, r_idx])
        try:
            V_new, _, _, _ = np.linalg.lstsq(UW, T2, rcond=None)
            V = V_new
        except:
            pass
        
        # Update W: T_3 @ (U ⊙ V)^+
        T3 = T.transpose(2, 0, 1).reshape(shape[2], -1)
        UV = np.zeros((shape[0] * shape[1], r))
        for r_idx in range(r):
            UV[:, r_idx] = np.kron(U[:, r_idx], V[:, r_idx])
        try:
            W_new, _, _, _ = np.linalg.lstsq(UV, T3, rcond=None)
            W = W_new
        except:
            pass
        
        # Check convergence
        T_recon = np.zeros(shape)
        for r_idx in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        error = np.linalg.norm(T_recon - T) / (np.linalg.norm(T) + 1e-15)
        if error < tol:
            break
    
    T_recon = np.zeros(shape)
    for r_idx in range(r):
        T_recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
    error = np.linalg.norm(T_recon - T) / (np.linalg.norm(T) + 1e-15)
    if error < tol:
        return U, V, W, True
    return None, None, None, False

def _to_list(arr):
    return [[float(x) for x in row] for row in arr]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    T = _build_target_tensor(n, m, p)
    
    # Try ranks from lower bound upward
    for r in range(23, 55):
        if time.time() > deadline:
            break
        
        for seed in range(10):
            if time.time() > deadline:
                break
            
            U, V, W, success = _als_decompose(T, r, max_iter=80, seed=seed + 100)
            if success:
                if _verify(U, V, W, n, m, p, tol=1e-10):
                    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
