"""Rewritten program: search for a low-rank exact tensor decomposition of 3x5 * 5x6."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]
n, m, p = DIMENSIONS
N = n * m  # 15
M = m * p  # 30
P = p * n  # 18
SCHOOLBOOK_RANK = n * m * p  # 90

def _schoolbook():
    """Return the schoolbook decomposition (rank 90)."""
    rank = SCHOOLBOOK_RANK
    U = np.zeros((N, rank), dtype=float)
    V = np.zeros((M, rank), dtype=float)
    W = np.zeros((P, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _to_lists(result):
    return {
        "U": [[float(x) for x in row] for row in result["U"]],
        "V": [[float(x) for x in row] for row in result["V"]],
        "W": [[float(x) for x in row] for row in result["W"]],
    }

def _compute_tensor(U, V, W):
    """Compute the tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]"""
    r = U.shape[1]
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr, optimize=True)
    return T

def _check_exact(U, V, W, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    T = _compute_tensor(U, V, W)
    target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, target, atol=tol, rtol=0)

def _objective(U, V, W):
    """Return sum of squared errors: ||T - 1||^2"""
    T = _compute_tensor(U, V, W)
    target = np.ones((n, m, p), dtype=float)
    diff = T - target
    return np.sum(diff * diff)

def _gradients(U, V, W):
    """Compute gradients of objective w.r.t. U, V, W."""
    r = U.shape[1]
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    
    T = _compute_tensor(U, V, W)
    target = np.ones((n, m, p), dtype=float)
    residual = T - target  # shape (n, m, p)
    
    # dL/dU[i,j,c] = sum_{k} residual[i,j,k] * Vr[j,k,c] * Wr[k,i,c]
    # dL/dV[j,k,c] = sum_{i} residual[i,j,k] * Ur[i,j,c] * Wr[k,i,c]
    # dL/dW[k,i,c] = sum_{j} residual[i,j,k] * Ur[i,j,c] * Vr[j,k,c]
    
    # For U: dU[i,j,c] = sum_k residual[i,j,k] * Vr[j,k,c] * Wr[k,i,c]
    # Reshape: residual is (n,m,p), Vr is (m,p,r), Wr is (p,n,r)
    # We need for each (i,j,c): sum_k residual[i,j,k] * Vr[j,k,c] * Wr[k,i,c]
    
    # Let's compute via einsum:
    # dU[i,j,c] = sum_k residual[i,j,k] * Vr[j,k,c] * Wr[k,i,c]
    # = einsum('ijk,jkc,kic->ijc', residual, Vr, Wr)
    
    dUr = np.einsum('ijk,jkc,kic->ijc', residual, Vr, Wr, optimize=True)
    
    # dV[j,k,c] = sum_i residual[i,j,k] * Ur[i,j,c] * Wr[k,i,c]
    dVr = np.einsum('ijk,ijc,kic->jkc', residual, Ur, Wr, optimize=True)
    
    # dW[k,i,c] = sum_j residual[i,j,k] * Ur[i,j,c] * Vr[j,k,c]
    dWr = np.einsum('ijk,ijc,jkc->kic', residual, Ur, Vr, optimize=True)
    
    dU = dUr.reshape(N, r)
    dV = dVr.reshape(M, r)
    dW = dWr.reshape(P, r)
    
    return dU, dV, dW

def _optimize(U, V, W, max_iter=500, lr_init=0.01, tol=1e-10, time_limit=30.0, start_time=None):
    """Gradient descent optimization with adaptive learning rate."""
    if start_time is None:
        start_time = time.time()
    
    obj = _objective(U, V, W)
    if obj < tol:
        return U, V, W, obj
    
    lr = lr_init
    best_U, best_V, best_W, best_obj = U.copy(), V.copy(), W.copy(), obj
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if obj < tol:
            break
        
        dU, dV, dW = _gradients(U, V, W)
        
        # Adaptive learning rate: reduce if objective increases
        U_new = U - lr * dU
        V_new = V - lr * dV
        W_new = W - lr * dW
        
        obj_new = _objective(U_new, V_new, W_new)
        
        if obj_new < obj:
            U, V, W = U_new, V_new, W_new
            obj = obj_new
            if obj < best_obj:
                best_U, best_V, best_W, best_obj = U.copy(), V.copy(), W.copy(), obj
            # Increase lr slightly if improving well
            if it % 10 == 0 and obj < 0.1:
                lr *= 1.05
        else:
            lr *= 0.5
            if lr < 1e-10:
                break
        
        # Check convergence
        if np.linalg.norm(dU) < 1e-8 and np.linalg.norm(dV) < 1e-8 and np.linalg.norm(dW) < 1e-8:
            break
    
    return best_U, best_V, best_W, best_obj

def _try_rank(r, rng, time_limit_per_attempt=15.0, max_attempts=5):
    """Try to find a rank-r decomposition."""
    best_obj = np.inf
    best_U = best_V = best_W = None
    
    for attempt in range(max_attempts):
        if time.time() > time_limit_per_attempt * (attempt + 1):
            break
        
        # Random initialization
        U = rng.standard_normal((N, r))
        V = rng.standard_normal((M, r))
        W = rng.standard_normal((P, r))
        
        # Scale
        U /= np.sqrt(np.sum(U**2))
        V /= np.sqrt(np.sum(V**2))
        W /= np.sqrt(np.sum(W**2))
        
        # Optimize
        U_opt, V_opt, W_opt, obj = _optimize(U, V, W, max_iter=300, lr_init=0.01, tol=1e-12, time_limit=time_limit_per_attempt)
        
        if obj < best_obj:
            best_obj = obj
            best_U, best_V, best_W = U_opt, V_opt, W_opt
        
        if best_obj < 1e-10:
            break
    
    if best_U is not None and best_obj < 1e-8:
        return {"U": best_U, "V": best_V, "W": best_W}, best_obj
    
    return None, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = SCHOOLBOOK_RANK
    
    rng = np.random.default_rng(42)
    
    # Try ranks from 89 down to 60
    for r in range(89, 59, -1):
        if time.time() > deadline:
            break
        
        result, obj = _try_rank(r, rng, time_limit_per_attempt=10.0, max_attempts=3)
        
        if result is not None:
            # Verify exactness
            if _check_exact(result["U"], result["V"], result["W"], tol=1e-8):
                best = result
                best_rank = r
                break
    
    # If we found something better, return it; otherwise return schoolbook
    if best_rank < SCHOOLBOOK_RANK:
        return _to_lists(best)
    
    return _to_lists(best)
