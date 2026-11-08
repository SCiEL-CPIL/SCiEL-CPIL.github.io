import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def _build_tensor(n, m, p):
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-8):
    """Verify that sum_r U[:,r]*V[:,r]*W[:,r] == T"""
    dim_U = U.shape[0]
    dim_V = V.shape[0]
    dim_W = W.shape[0]
    # Compute tensor product sum
    reconstructed = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for r in range(U.shape[1]):
        reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(reconstructed, T, atol=tol, rtol=tol)

def _als_decompose(T, rank, n_iter=200, tol=1e-10, seed=42):
    """ALS decomposition for the matrix multiplication tensor"""
    rng = np.random.RandomState(seed)
    dim_U, dim_V, dim_W = T.shape
    
    # Initialize with random Gaussian
    U = rng.randn(dim_U, rank) * 0.1
    V = rng.randn(dim_V, rank) * 0.1
    W = rng.randn(dim_W, rank) * 0.1
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(n_iter):
        # Update U: fix V and W
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed V, W: T[a] = sum_r U[a,r] * (V*W)[r] where (V*W) is outer product per r
        # Actually: T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
        # So: reshape T to (dim_U, dim_V*dim_W), and for each r, the coefficient vector is outer(V[:,r], W[:,r])
        # We want: T_reshaped = U @ M where M[r, (b,c)] = V[b,r]*W[c,r]
        
        M = np.zeros((rank, dim_V * dim_W))
        for r in range(rank):
            M[r, :] = np.outer(V[:, r], W[:, r]).flatten()
        
        T_reshaped = T.reshape(dim_U, dim_V * dim_W)
        U_new = np.linalg.lstsq(M.T, T_reshaped.T, rcond=None)[0].T
        
        # Update V
        M2 = np.zeros((rank, dim_U * dim_W))
        for r in range(rank):
            M2[r, :] = np.outer(U_new[:, r], W[:, r]).flatten()
        T_reshaped2 = T.transpose(1, 0, 2).reshape(dim_V, dim_U * dim_W)
        V_new = np.linalg.lstsq(M2.T, T_reshaped2.T, rcond=None)[0].T
        
        # Update W
        M3 = np.zeros((rank, dim_U * dim_V))
        for r in range(rank):
            M3[r, :] = np.outer(U_new[:, r], V_new[:, r]).flatten()
        T_reshaped3 = T.transpose(2, 0, 1).reshape(dim_W, dim_U * dim_V)
        W_new = np.linalg.lstsq(M3.T, T_reshaped3.T, rcond=None)[0].T
        
        # Compute error
        error = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
        for r in range(rank):
            error += np.einsum('a,b,c->abc', U_new[:, r], V_new[:, r], W_new[:, r])
        error -= T
        err_val = np.linalg.norm(error)
        
        if err_val < best_error:
            best_error = err_val
            best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
        
        U, V, W = U_new, V_new, W_new
        
        if err_val < tol:
            break
    
    return best_U, best_V, best_W, best_error

def _try_rank(T, rank, n_trials=50, time_limit=140.0, start_time=None):
    """Try to find a valid rank-r decomposition"""
    if start_time is None:
        start_time = time.time()
    
    dim_U, dim_V, dim_W = T.shape
    
    for trial in range(n_trials):
        if time.time() - start_time > time_limit - 10.0:
            break
        
        seed = trial * 1000 + 42
        U, V, W, err = _als_decompose(T, rank, n_iter=100, tol=1e-12, seed=seed)
        
        if err < 1e-8:
            # Verify exactly
            if _verify(U, V, W, T, tol=1e-10):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    T = _build_tensor(n, m, p)
    
    start_time = time.time()
    time_limit = 140.0
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_U, schoolbook_rank))
    V_sb = np.zeros((dim_V, schoolbook_rank))
    W_sb = np.zeros((dim_W, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    for rank in [19, 20, 21, 22, 23]:
        if time.time() - start_time > time_limit - 30.0:
            break
        
        result = _try_rank(T, rank, n_trials=30, time_limit=time_limit, start_time=start_time)
        
        if result is not None:
            U, V, W = result
            # Convert to lists
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank
            break
    
    return best_result
