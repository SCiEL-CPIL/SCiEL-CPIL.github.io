import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """7 terms for 2x2x2 Strassen."""
    return [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

def _build_kronecker_strassen_444():
    """Kronecker-embedded Strassen for 4x4x4 (rank 49)."""
    n, m, p = 4, 4, 4
    U_cols, V_cols, W_cols = [], [], []

    def add(u, v, w):
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

    def ui(i, j): return i * m + j
    def vi(j, k): return j * p + k
    def wi(k, i): return k * n + i

    st = _strassen_2x2x2_terms()
    for (u1s, v1s, w1s) in st:
        for (u2s, v2s, w2s) in st:
            u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
            for (a, va) in u1s:
                i1, j1 = a // 2, a % 2
                for (b, vb) in u2s:
                    i2, j2 = b // 2, b % 2
                    u[ui(i1 * 2 + i2, j1 * 2 + j2)] = va * vb
            for (a, va) in v1s:
                j1, k1 = a // 2, a % 2
                for (b, vb) in v2s:
                    j2, k2 = b // 2, b % 2
                    v[vi(j1 * 2 + j2, k1 * 2 + k2)] = va * vb
            for (a, va) in w1s:
                k1, i1 = a // 2, a % 2
                for (b, vb) in w2s:
                    k2, i2 = b // 2, b % 2
                    w[wi(k1 * 2 + k2, i1 * 2 + i2)] = va * vb
            add(u, v, w)

    return np.array(U_cols).T, np.array(V_cols).T, np.array(W_cols).T

def _build_optimized_baseline(n=4, m=4, p=5):
    """
    Builds the rank-50 baseline:
    49 terms for 4x4x4 (Strassen) + 1 term for 4x4x1 (rank 1).
    """
    # Get the 49 terms for 4x4x4
    U444, V444, W444 = _build_kronecker_strassen_444()
    
    # Initialize arrays for 4x4x5
    U = np.zeros((n * m, 50))
    V = np.zeros((m * p, 50))
    W = np.zeros((p * n, 50))
    
    # Copy the 49 terms
    # Note: U444 has shape (16, 49), V444 (20, 49), W444 (20, 49)
    # We need to map indices correctly.
    # U444 indices: i*4+j
    # V444 indices: j*4+k (k in 0..3)
    # W444 indices: k*4+i (k in 0..3)
    
    U[:, :49] = U444
    # V444 has k in 0..3. V has k in 0..4.
    # So V[:, :49] should have non-zeros in rows j*4+k for k=0..3.
    V[:, :49] = V444
    # W444 has k in 0..3. W has k in 0..4.
    W[:, :49] = W444
    
    # Add the 1 term for k=4
    # U[i, j] = 1 for all i, j in 0..3
    U[0:16, 49] = 1.0
    # V[j, 4] = 1 for all j in 0..3
    # Row index for V is j*p + k. Here k=4.
    # So rows are j*5 + 4 for j=0..3 -> 4, 9, 14, 19.
    V[4, 49] = 1.0
    V[9, 49] = 1.0
    V[14, 49] = 1.0
    V[19, 49] = 1.0
    # W[4, i] = 1 for all i in 0..3
    # Row index for W is k*n + i. Here k=4.
    # So rows are 4*4 + i = 16 + i for i=0..3 -> 16, 17, 18, 19.
    W[16, 49] = 1.0
    W[17, 49] = 1.0
    W[18, 49] = 1.0
    W[19, 49] = 1.0
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-7):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            row_u = U[i * m + j]
            for k in range(p):
                s = np.dot(row_u * V[j * p + k], W[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_int(M, tol=1e-8):
    M = np.asarray(M, dtype=np.float64)
    return bool(np.all(np.abs(M * 2 - np.round(M * 2)) <= tol))

def _round_half(M):
    return np.round(np.asarray(M, dtype=np.float64) * 2) / 2.0

def _build_schoolbook(n=4, m=4, p=5):
    rank = n * m * p
    U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _to_list(M):
    return [[float(x) for x in row] for row in np.asarray(M)]

def _null_space_prune(U, V, W, n, m, p, max_iter=1000, tol=1e-8):
    """
    Uses null space of the tensor flattening to find linear dependencies
    and prune terms.
    """
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    rank = U.shape[1]
    
    # Flatten tensor vectors
    vecs = []
    for r in range(rank):
        vec = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                u_val = U[i*m+j, r]
                if u_val == 0: continue
                for k in range(p):
                    v_val = V[j*p+k, r]
                    if v_val == 0: continue
                    w_val = W[k*n+i, r]
                    if w_val == 0: continue
                    vec[i*m*p + j*p + k] = u_val * v_val * w_val
        vecs.append(vec)
        
    A = np.array(vecs).T # Shape (n*m*p, rank)
    t = np.sum(A, axis=1) # The target tensor vector
    
    N = null_space(A)
    
    if N.shape[1] == 0:
        return U, V, W
        
    alpha = np.ones(rank)
    
    best_alpha = alpha.copy()
    best_active_count = rank
    
    for restart in range(5):
        alpha = np.ones(rank)
        
        improved = True
        while improved:
            improved = False
            best_k = -1
            best_new_alpha = None
            best_active = 0
            
            for k in range(rank):
                if alpha[k] == 0:
                    continue
                n_k = N[:, k]
                norm_sq = np.dot(n_k, n_k)
                if norm_sq < 1e-12:
                    continue
                z = -n_k / norm_sq
                new_alpha = 1 + N @ z
                
                valid = True
                active = 0
                for r in range(rank):
                    if abs(new_alpha[r]) > 1e-8:
                        if not _is_half_int(np.array([[new_alpha[r]]])):
                            valid = False
                            break
                        active += 1
                    else:
                        if abs(new_alpha[r]) > 1e-4:
                            valid = False
                            break
                        active += 0
                
                if valid and active > best_active:
                    best_active = active
                    best_k = k
                    best_new_alpha = new_alpha.copy()
            
            if best_k >= 0 and best_new_alpha is not None:
                alpha = best_new_alpha
                alpha[np.abs(alpha) < 1e-6] = 0
                improved = True
        
        active_count = np.sum(np.abs(alpha) > 1e-8)
        if active_count < best_active_count:
            best_active_count = active_count
            best_alpha = alpha.copy()
    
    alpha = best_alpha
    
    final_U = U.copy()
    final_V = V.copy()
    final_W = W.copy()
    
    active = []
    for r in range(rank):
        if abs(alpha[r]) > 1e-8:
            a = alpha[r]
            success = False
            
            final_U[:, r] = U[:, r] * a
            if _is_half_int(final_U[:, r]):
                active.append(r)
                success = True
            else:
                final_U[:, r] = U[:, r]
            
            if not success:
                final_V[:, r] = V[:, r] * a
                if _is_half_int(final_V[:, r]):
                    active.append(r)
                    success = True
                else:
                    final_V[:, r] = V[:, r]
            
            if not success:
                final_W[:, r] = W[:, r] * a
                if _is_half_int(final_W[:, r]):
                    active.append(r)
                    success = True
                else:
                    final_W[:, r] = W[:, r]
            
            if not success and a > 0:
                s = np.sqrt(a)
                final_U[:, r] = U[:, r] * s
                final_V[:, r] = V[:, r] * s
                if _is_half_int(final_U[:, r]) and _is_half_int(final_V[:, r]):
                    active.append(r)
                    success = True
                else:
                    final_U[:, r] = U[:, r]
                    final_V[:, r] = V[:, r]
            
            if not success and a > 0:
                s = np.sqrt(a)
                final_U[:, r] = U[:, r] * s
                final_W[:, r] = W[:, r] * s
                if _is_half_int(final_U[:, r]) and _is_half_int(final_W[:, r]):
                    active.append(r)
                    success = True
                else:
                    final_U[:, r] = U[:, r]
                    final_W[:, r] = W[:, r]
            
            if not success and a > 0:
                s = np.sqrt(a)
                final_V[:, r] = V[:, r] * s
                final_W[:, r] = W[:, r] * s
                if _is_half_int(final_V[:, r]) and _is_half_int(final_W[:, r]):
                    active.append(r)
                    success = True
                else:
                    final_V[:, r] = V[:, r]
                    final_W[:, r] = W[:, r]
            
            if not success:
                pass
        else:
            pass
    
    if len(active) >= rank:
        return U, V, W
        
    new_U = final_U[:, active]
    new_V = final_V[:, active]
    new_W = final_W[:, active]
    
    if _verify(new_U, new_V, new_W, n, m, p):
        return new_U, new_V, new_W
        
    return U, V, W

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    n, m, p = DIMENSIONS

    best = None

    def consider(U, V, W, label=""):
        nonlocal best
        if time.time() > deadline:
            return
        U = _round_half(U); V = _round_half(V); W = _round_half(W)
        if not _verify(U, V, W, n, m, p):
            return
        if not (_is_half_int(U) and _is_half_int(V) and _is_half_int(W)):
            return
        r = U.shape[1]
        if best is None or r < best[0]:
            best = (r, U, V, W)

    # 1) Optimized baseline (Rank 50)
    try:
        U, V, W = _build_optimized_baseline(n, m, p)
        consider(U, V, W, "optimized_baseline")
        
        # Try pruning
        if time.time() < deadline - 20:
            U2, V2, W2 = _null_space_prune(U, V, W, n, m, p)
            consider(U2, V2, W2, "pruned_baseline")
    except Exception:
        pass

    # 2) Fallback schoolbook
    try:
        U, V, W = _build_schoolbook(n, m, p)
        consider(U, V, W, "schoolbook")
    except Exception:
        pass

    if best is None:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    _, U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
