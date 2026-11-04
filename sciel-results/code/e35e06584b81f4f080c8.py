import numpy as np
import time
import random
import itertools

_START = time.time()
_DEADLINE = _START + 110

def _verify(U, V, W, tol=1e-10):
    T = np.einsum('ar,br,cr->abc', U, V, W)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    valid_mask = np.zeros((9, 9, 9), dtype=bool)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                valid_mask[a, b, c] = True
    invalid = T[~valid_mask]
    if np.max(np.abs(invalid)) > tol:
        return False
    return True

def _als_decompose(rank, max_iters=200, tol=1e-9, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    U = rng.randn(9, rank)
    V = rng.randn(9, rank)
    W = rng.randn(9, rank)
    
    M = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                M[a, b, c] = 1.0
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 15:
            break
            
        # Update U
        B_mat = np.zeros((81, rank))
        for r in range(rank):
            outer_vw = np.outer(V[:, r], W[:, r])
            B_mat[:, r] = outer_vw.flatten()
        M_mat = M.transpose(0, 1, 2).reshape(9, 81)
        try:
            U_new, _, _, _ = np.linalg.lstsq(B_mat.T, M_mat.T, rcond=None)
            U = U_new.T
        except:
            pass
            
        # Update V
        M_mat2 = M.transpose(1, 0, 2).reshape(9, 81)
        B_mat2 = np.zeros((81, rank))
        for r in range(rank):
            outer_uw = np.outer(U[:, r], W[:, r])
            B_mat2[:, r] = outer_uw.flatten()
        try:
            V_new, _, _, _ = np.linalg.lstsq(B_mat2.T, M_mat2.T, rcond=None)
            V = V_new.T
        except:
            pass
            
        # Update W
        M_mat3 = M.transpose(2, 0, 1).reshape(9, 81)
        B_mat3 = np.zeros((81, rank))
        for r in range(rank):
            outer_uv = np.outer(U[:, r], V[:, r])
            B_mat3[:, r] = outer_uv.flatten()
        try:
            W_new, _, _, _ = np.linalg.lstsq(B_mat3.T, M_mat3.T, rcond=None)
            W = W_new.T
        except:
            pass
            
        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - M)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < tol:
            break
            
    U_r = np.round(best_U * 2) / 2
    V_r = np.round(best_V * 2) / 2
    W_r = np.round(best_W * 2) / 2
    
    if _verify(U_r, V_r, W_r):
        return U_r, V_r, W_r
    return None

def _try_known_23():
    # Attempt to use a known 23-term decomposition if one can be reliably constructed.
    # Since deriving it is error-prone, we rely on the ALS fallback.
    # However, we can try a few specific structured inits.
    pass

def construct():
    # Try ranks from 23 to 26
    for rank in [23, 24, 25, 26]:
        if time.time() > _DEADLINE - 10:
            break
        # Try multiple seeds for each rank
        for seed in range(10):
            if time.time() > _DEADLINE - 10:
                break
            result = _als_decompose(rank, max_iters=150, tol=1e-9, seed=seed)
            if result:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to 27
    U_fb = np.zeros((9, 27), dtype=float)
    V_fb = np.zeros((9, 27), dtype=float)
    W_fb = np.zeros((9, 27), dtype=float)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_fb[i*3+j, col] = 1.0
                V_fb[j*3+k, col] = 1.0
                W_fb[k*3+i, col] = 1.0
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
