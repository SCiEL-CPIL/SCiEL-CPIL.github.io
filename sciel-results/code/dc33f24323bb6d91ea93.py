"""Program for matrix multiplication tensor decomposition of (3,3,8)."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 15.0


def _schoolbook_decomposition(n, m, p):
    """Build the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _tensor_residual(U, V, W, n, m, p):
    """Compute the residual of the tensor decomposition."""
    U_arr = np.asarray(U, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    
    residual = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    return residual - target


def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    residual = _tensor_residual(U, V, W, n, m, p)
    return np.max(np.abs(residual)) < tol


def _optimize_rank(U_init, V_init, W_init, n, m, p, target_rank, max_time=25.0, tol=1e-8):
    """Optimize U, V, W to achieve target_rank with minimal residual."""
    start_time = time.time()
    
    U = U_init[:, :target_rank].copy()
    V = V_init[:, :target_rank].copy()
    W = W_init[:, :target_rank].copy()
    
    lr = 0.005
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    while time.time() - start_time < max_time:
        residual = _tensor_residual(U, V, W, n, m, p)
        res_norm = np.linalg.norm(residual)
        
        if res_norm < best_residual:
            best_residual = res_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if res_norm < tol:
            return best_U, best_V, best_W, target_rank
        
        grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
        grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
        grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        new_res = _tensor_residual(U, V, W, n, m, p)
        new_norm = np.linalg.norm(new_res)
        if new_norm > res_norm * 1.5:
            lr *= 0.5
            U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
        elif new_norm < res_norm * 0.9:
            lr = min(lr * 1.1, 0.05)
    
    return best_U, best_V, best_W, target_rank


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        pass
    
    remaining = DEADLINE_SECONDS - SAFETY_MARGIN - (time.time() - start_time)
    
    target_ranks = [64, 60, 56, 52, 48, 44]
    
    for tr in target_ranks:
        if time.time() - start_time > DEADLINE_SECONDS - SAFETY_MARGIN:
            break
        
        time_budget = min(20.0, remaining / 2.0)
        if time_budget < 3.0:
            break
        remaining -= time_budget
        
        U_try, V_try, W_try, rank_ach = _optimize_rank(
            U, V, W, n, m, p, tr, max_time=time_budget
        )
        
        if _verify_decomposition(U_try, V_try, W_try, n, m, p):
            if rank_ach < best_rank:
                best_rank = rank_ach
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
    
    # Round to half-integers
    U_final = np.round(best_U * 2) / 2
    V_final = np.round(best_V * 2) / 2
    W_final = np.round(best_W * 2) / 2
    
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        best_U, best_V, best_W = U_final, V_final, W_final
    else:
        # Try without rounding
        pass
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
