import numpy as np
import time
import random

_START_TIME = time.time()
_TIMEOUT = 130

def _get_tensor_223():
    n, m, p = 2, 2, 3
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _search_rank_10_223():
    T = _get_tensor_223()
    n, m, p = 2, 2, 3
    rank = 10
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Normalize
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    
    for step in range(500):
        if time.time() - _START_TIME > _TIMEOUT - 20: break
        
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.sum((T - T_recon)**2)
        if err < 1e-10:
            return U, V, W
            
        # Gradient
        # dT_recon/dU[i,r] = V[j,r] * W[k,r]
        # grad_U[i,r] = 2 * sum_{j,k} (T[i,j,k] - T_recon[i,j,k]) * V[j,r] * W[k,r]
        diff = T - T_recon
        grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
        grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
        grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
        
        # Simple gradient descent
        lr = 0.01
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Re-normalize to prevent divergence
        U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-9
        V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-9
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9

    return None

def _build_rank_30():
    res = _search_rank_10_223()
    if res is None:
        return None
    U3, V3, W3 = res
    # Construct 2x2x9 from 3 blocks of 2x2x3
    U9 = np.zeros((4, 30))
    V9 = np.zeros((18, 30))
    W9 = np.zeros((18, 30))
    
    for block in range(3):
        # U is the same
        U9[:, block*10:(block+1)*10] = U3
        
        # V and W are shifted
        # V3 has shape (6, 10). V9 has shape (18, 30).
        # Block 0: V9[0:6, 0:10] = V3
        # Block 1: V9[6:12, 10:20] = V3
        # Block 2: V9[12:18, 20:30] = V3
        V9[block*6:(block+1)*6, block*10:(block+1)*10] = V3
        W9[block*6:(block+1)*6, block*10:(block+1)*10] = W3
        
    return U9.tolist(), V9.tolist(), W9.tolist()

def _build_rank_32(n, m, p):
    # ... (keep existing code) ...
    pass

def construct():
    n, m, p = 2, 2, 9
    
    # Try rank 30
    res = _build_rank_30()
    if res:
        U, V, W = res
        if _validate_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # Fallback to 32
    U, V, W = _build_rank_32(n, m, p)
    if _validate_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
        
    # Fallback to 36
    # ...
