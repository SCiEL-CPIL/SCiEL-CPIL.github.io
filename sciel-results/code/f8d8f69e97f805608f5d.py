import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 10]
_DEADLINE = 120.0

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return bool(np.allclose(recon, T, atol=1e-9, rtol=0.0))

_STRASSEN = [
    ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
    ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
    ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
    ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
    ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),
    ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
    ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
]

def _emit_strassen(U, V, W, n, p, base_col, c0, c1):
    for t in range(7):
        col = base_col + t
        u_side, v_side, w_side = _STRASSEN[t]
        for u_row, coeff in u_side:
            U[u_row][col] += coeff
        for v_koff, coeff in v_side:
            j = v_koff // 2
            k = c0 if v_koff % 2 == 0 else c1
            V[j * p + k][col] += coeff
        for w_koff, coeff in w_side:
            i = w_koff // 2
            k = c0 if w_koff % 2 == 0 else c1
            W[k * n + i][col] += coeff

def _build_full_tensor(n, m, p):
    """Build the full matrix multiplication tensor T[i,j,k]"""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _try_rank_r(T, r, max_iters=50, seed=42, deadline=None):
    """
    Try to find a rank-r decomposition using ALS with random restarts.
    Returns (U, V, W) as numpy arrays if successful, else None.
    """
    np.random.seed(seed)
    N, P, Q = T.shape  # 4, 20, 20
    
    best_error = float('inf')
    best_solution = None
    
    for attempt in range(20):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(N, r) * 0.1
        V = np.random.randn(P, r) * 0.1
        W = np.random.randn(Q, r) * 0.1
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Update U using least squares
            A = np.zeros((P * Q, r))
            for col in range(r):
                A[:, col] = (V[:, col] * W[:, col][:, None]).flatten()
            
            try:
                U = np.linalg.lstsq(A.T, T.reshape(N, P * Q).T, rcond=None)[0].T
            except:
                continue
            
            # Update V
            B = np.zeros((N * Q, r))
            for col in range(r):
                B[:, col] = (U[:, col] * W[:, col][:, None]).flatten()
            try:
                V = np.linalg.lstsq(B.T, T.reshape(P, N * Q).T, rcond=None)[0].T
            except:
                continue
            
            # Update W
            C = np.zeros((N * P, r))
            for col in range(r):
                C[:, col] = (U[:, col] * V[:, col][:, None]).flatten()
            try:
                W = np.linalg.lstsq(C.T, T.reshape(Q, N * P).T, rcond=None)[0].T
            except:
                continue
            
            # Check convergence
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(recon - T)
            if error < 1e-10:
                break
        
        if deadline is not None and time.time() > deadline:
            break
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        recon = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
        error = np.linalg.norm(recon - T)
        
        if error < best_error:
            best_error = error
            best_solution = (U_rounded, V_rounded, W_rounded)
        
        if error < 1e-6:
            break
    
    if best_error < 1e-6:
        return best_solution
    return None

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 2, 2, 10
    deadline = t0 + 100  # Leave 20s margin
    
    T = _build_full_tensor(n, m, p)
    
    # Try ranks from 34 down to 25
    for r in range(34, 24, -1):
        if time.time() > deadline:
            break
        
        result = _try_rank_r(T, r, max_iters=50, seed=42 + r, deadline=deadline)
        
        if result is not None:
            U_rounded, V_rounded, W_rounded = result
            U = U_rounded.tolist()
            V = V_rounded.tolist()
            W = W_rounded.tolist()
            
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: safe rank-35 baseline
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for b in range(5):
        _emit_strassen(U, V, W, n, p, b * 7, 2 * b, 2 * b + 1)

    if not _verify(U, V, W, n, m, p):
        raise RuntimeError("rank-35 fallback failed verification")
    return {"U": U, "V": V, "W": W}
