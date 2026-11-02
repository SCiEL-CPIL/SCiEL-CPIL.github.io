import time, random, math
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 138.0
_T0 = time.time()
_RNG = random.Random(1234567)

def _time_left():
    return DEADLINE - (time.time() - _T0)

def _round_half(A):
    return (2.0 * np.asarray(A, dtype=float)).round() / 2.0

def _compute_tensor(U, V, W):
    n, m, p = DIMENSIONS
    r = U.shape[1]
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    ref = np.zeros((p * n, n * m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ref[k * n + i, j * m + k] = T[i, j, k]
    return ref

def _validate(U, V, W):
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float); W = np.asarray(W, dtype=float)
    n, m, p = DIMENSIONS
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    T = _compute_tensor(U, V, W)
    ref = np.zeros((n * p, n * m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ref[k * n + i, j * m + k] = 1.0
    
    if not np.allclose(T, ref, rtol=0.0, atol=1e-6):
        return False
    return bool(np.allclose(U, _round_half(U), rtol=0.0, atol=1e-4)) and \
           bool(np.allclose(V, _round_half(V), rtol=0.0, atol=1e-4)) and \
           bool(np.allclose(W, _round_half(W), rtol=0.0, atol=1e-4))

def _loss(U, V, W, ref):
    T = _compute_tensor(U, V, W)
    return float(np.sum((T - ref) ** 2))

def _local_search(U, V, W, ref, target, budget):
    n, m, p = DIMENSIONS
    r = U.shape[1]
    U = U.astype(float).copy(); V = V.astype(float).copy(); W = W.astype(float).copy()
    cur = _loss(U, V, W, ref)
    if cur <= target:
        return U, V, W
    t = 0.05
    step = 0
    best = (U.copy(), V.copy(), W.copy(), cur)
    while cur > target and step < budget and _time_left() > 3.0:
        M = _RNG.choice([0, 1, 2])
        A = (U, V, W)[M]
        idx = tuple(_RNG.randrange(d) for d in A.shape)
        old = A[idx]
        A[idx] = old + _RNG.gauss(0.0, t)
        nl = _loss(U, V, W, ref)
        if nl < cur:
            cur = nl
            if cur < best[3]:
                best = (U.copy(), V.copy(), W.copy(), cur)
            t = min(t * 1.02, 0.1)
        else:
            A[idx] = old
            t = max(t * 0.98, 0.01)
        step += 1
    return best[0], best[1], best[2]

def _ref_tensor():
    n, m, p = DIMENSIONS
    ref = np.zeros((p * n, n * m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ref[k * n + i, j * m + k] = 1.0
    return ref

def construct():
    n, m, p = DIMENSIONS
    ref = _ref_tensor()
    best = None

    # Try rank 19 (known optimal for 2x3x4 matmul? Actually 2x3x4 matmul rank is 19? 
    # Strassen for 2x2 is 7. For 2x3x4, it's a specific case.
    # Schoolbook is 24.
    # Let's try 19, 20, 21...
    
    ranks_to_try = [19, 20, 21, 22, 23, 24]
    
    for rank in ranks_to_try:
        if _time_left() < 10.0:
            break
            
        # Initialize
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Local search
        U, V, W = _local_search(U, V, W, ref, 1e-6, 2000)
        
        # Round to half-integers
        U_r = _round_half(U)
        V_r = _round_half(V)
        W_r = _round_half(W)
        
        if _validate(U_r, V_r, W_r):
            best = (U_r, V_r, W_r)
            break
            
        # If we find a valid solution with higher rank, keep it as fallback
        if best is None and _validate(U_r, V_r, W_r):
             best = (U_r, V_r, W_r)

    # Fallback: Schoolbook (rank 24)
    if best is None:
        rank = n * m * p
        U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        best = (U, V, W)

    U, V, W = best
    U = _round_half(U); V = _round_half(V); W = _round_half(W)
    if not _validate(U, V, W):
        # Last resort
        rank = n * m * p
        U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
