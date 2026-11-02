"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE = 150  # seconds wall-clock limit
SAFETY_MARGIN = 5  # seconds to return before deadline

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Check if U, V, W (each of shape (rows, rank)) form an exact decomposition.
    """
    r = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(r):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        u_mat = uc.reshape(n, m)
        v_mat = vc.reshape(m, p)
        w_mat = wc.reshape(p, n)
        for i in range(n):
            for j in range(m):
                T[i, j, :] += u_mat[i, j] * (v_mat[j, :] * w_mat[:, i])
    return np.allclose(T, 1.0, atol=tol)


def _round_and_check(U, V, W, n, m, p):
    """
    Try rounding U, V, W to integers or half-integers and verify.
    Returns (rank, U, V, W) if successful, else (None, None, None, None).
    """
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    if _verify(U_int, V_int, W_int, n, m, p, tol=1e-9):
        return U_int.shape[1], U_int, V_int, W_int
    
    U_half = np.round(U * 2) / 2.0
    V_half = np.round(V * 2) / 2.0
    W_half = np.round(W * 2) / 2.0
    if _verify(U_half, V_half, W_half, n, m, p, tol=1e-9):
        return U_half.shape[1], U_half, V_half, W_half
        
    return None, None, None, None


def _optimize_decomposition(n, m, p, rank, time_limit=30, max_iters=2000, lr=0.01):
    """
    Use gradient descent to minimize || T(U,V,W) - 1 ||_F^2.
    """
    start_time = time.time()
    
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        T = np.zeros((n, m, p))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            u_mat = uc.reshape(n, m)
            v_mat = vc.reshape(m, p)
            w_mat = wc.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += u_mat[i, j] * (v_mat[j, :] * w_mat[:, i])
        
        error = np.linalg.norm(T - 1.0)
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if error < 1e-6:
            break
            
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        E = T - 1.0
        
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            u_mat = uc.reshape(n, m)
            v_mat = vc.reshape(m, p)
            w_mat = wc.reshape(p, n)
            
            du_mat = np.zeros((n, m))
            dv_mat = np.zeros((m, p))
            dw_mat = np.zeros((p, n))
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = E[i, j, k]
                        du_mat[i, j] += e * v_mat[j, k] * w_mat[k, i]
                        dv_mat[j, k] += e * u_mat[i, j] * w_mat[k, i]
                        dw_mat[k, i] += e * u_mat[i, j] * v_mat[j, k]
                        
            dU[:, c] = du_mat.flatten()
            dV[:, c] = dv_mat.flatten()
            dW[:, c] = dw_mat.flatten()
            
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
    rank_found, U_r, V_r, W_r = _round_and_check(best_U, best_V, best_W, n, m, p)
    if rank_found:
        return rank_found, U_r, V_r, W_r
        
    return None, None, None, None


def _try_block_decomposition(n, m, p, max_rank=None, time_limit=30):
    """
    Try to find a lower-rank decomposition.
    """
    start_time = time.time()
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    if max_rank is None:
        max_rank = best_rank
        
    min_rank_to_try = max(n * m, m * p, p * n)
    
    ranks_to_try = list(range(min_rank_to_try, min(min_rank_to_try + 20, best_rank)))
    
    for rank in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
        if rank >= best_rank:
            break
            
        for attempt in range(3):
            if time.time() - start_time > time_limit:
                break
                
            r, U, V, W = _optimize_decomposition(n, m, p, rank, time_limit=5)
            if r is not None and r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
                
    return best_rank, best_U, best_V, best_W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    time_remaining = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
    if time_remaining > 5:
        rank, U, V, W = _try_block_decomposition(n, m, p, time_limit=time_remaining)
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
    
    U_list = [[float(U[i][j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i][j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i][j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
