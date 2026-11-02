"""Matrix multiplication 2x3x7: greedy rank-1 peeling + half-integer projection."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]


def _peel(n, m, p, max_rank):
    """Greedy rank-1 peeling via 2D SVD on the (p*n)-matricization.
    Returns list of (a, u, v, w) with orthonormal u,v,w and a = 3**(1/4)."""
    # Correct T3 construction: Block diagonal with n blocks of size p x m, all 1s.
    T3 = np.zeros((p * n, n * m))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T3[k * n + i, i * m + j] = 1.0

    # Correct T2 construction: Block diagonal with m blocks of size n x p, all 1s.
    T2 = np.zeros((n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k] = 1.0

    terms = []
    scale = 3.0 ** 0.25
    for r in range(max_rank):
        # SVD of current T3
        U, s, Vt = np.linalg.svd(T3, full_matrices=False)
        if len(s) == 0 or s[0] <= 1e-9:
            break
        # The leading singular vector of T3 corresponds to the 'u' factor (n*m space).
        # T3 has shape (p*n, n*m). Its columns are indexed by (i*m + j) -> u index.
        # Its rows are indexed by (k*n + i) -> w index.
        u_vec = Vt[0, :]            # length n*m
        w_vec = U[:, 0]             # length p*n
        # Residual: T3 -= scale * w_vec outer u_vec
        T3 = T3 - scale * np.outer(w_vec, u_vec)

        # Reconstruct v from T2: T2[i*m+j, j*p+k] = 1.
        # T2 = sum a * u outer v.  => v = (1/a) * T2 u  (since U^T U = I).
        v_vec = T2 @ u_vec / scale

        # Normalize (should already be unit)
        nu = np.linalg.norm(u_vec)
        nv = np.linalg.norm(v_vec)
        nw = np.linalg.norm(w_vec)
        u_vec = u_vec / nu
        v_vec = v_vec / nv
        w_vec = w_vec / nw
        
        # Update T3 residual cleanly
        T3 = np.zeros((p * n, n * m))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    T3[k * n + i, i * m + j] = 1.0
        for (a2, u2, v2, w2) in terms:
            T3 = T3 - a2 * np.outer(w2, u2)
        T3 = T3 - scale * np.outer(w_vec, u_vec)

        terms.append((scale, u_vec, v_vec, w_vec))
    return terms


def _try_normalize_to_half_int(vec):
    """
    Try to scale the vector so that its non-zero entries become 1.0.
    Returns the scaled vector if successful (entries are 0 or 1), else None.
    """
    non_zero = vec[np.abs(vec) > 1e-9]
    if len(non_zero) == 0:
        return vec.copy()
    
    # Check if all non-zero entries are approximately equal
    mean_val = np.mean(non_zero)
    if np.std(non_zero) > 1e-6 * abs(mean_val):
        return None
        
    # Scale to make them 1
    scaled = vec / mean_val
    # Check if scaled is close to 0 or 1
    if not np.allclose(scaled[scaled > 0.5], 1.0, atol=1e-6):
        return None
    if not np.allclose(scaled[scaled < 0.5], 0.0, atol=1e-6):
        return None
        
    # Snap to 0 or 1
    snapped = np.where(scaled > 0.5, 1.0, 0.0)
    return snapped


def _project_half_integer(vec, tol=1e-3):
    """Project a vector to the nearest half-integer grid; return (projected, max_err)."""
    proj = np.round(2.0 * vec) / 2.0
    err = np.max(np.abs(proj - vec))
    return proj, err


def _is_half_integer(vec, tol=1e-6):
    scaled = np.round(2.0 * vec)
    return bool(np.all(np.abs(scaled - 2.0 * vec) < tol))


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W (as float arrays of shape (n*m, r), (m*p, r), (p*n, r))
    exactly decompose the multiplication tensor (within integer/half-integer exactness)."""
    r = U.shape[1]
    # Build T_3 = W @ U^T  (shape (p*n, n*m))
    T3 = W @ U.T
    ok = True
    for i in range(n):
        for k in range(p):
            for j in range(m):
                expected = 1.0
                actual = T3[k * n + i, i * m + j]
                if abs(actual - expected) > 1e-6:
                    ok = False
    # Also verify T_2 = U @ V^T
    T2 = U @ V.T
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = T2[i * m + j, j * p + k]
                if abs(actual - expected) > 1e-6:
                    ok = False
    # Verify T_1 = V @ W^T
    T1 = V @ W.T
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = T1[i * m + j, k * n + i]
                if abs(actual - expected) > 1e-6:
                    ok = False
    return ok


def _to_list(mat):
    out = []
    for row in mat:
        r = []
        for val in row:
            # Snap to half-integer
            hv = np.round(val * 2.0) / 2.0
            if abs(hv - round(hv)) < 1e-9:
                r.append(int(round(hv)))
            else:
                r.append(float(hv))
        out.append(r)
    return out


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110.0  # safety margin

    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try ranks from the theoretical minimum upward
    min_rank = 1
    max_rank = min(best_rank, 20)

    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
        try:
            terms = _peel(n, m, p, r)
            if len(terms) < r:
                continue
            
            # Build factor matrices
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Try to normalize vectors to 0/1 if possible
            success = True
            for c in range(r):
                a, u, v, w = terms[c]
                
                u_norm = _try_normalize_to_half_int(u)
                v_norm = _try_normalize_to_half_int(v)
                w_norm = _try_normalize_to_half_int(w)
                
                if u_norm is not None:
                    U[:, c] = u_norm
                else:
                    U[:, c], e1 = _project_half_integer(u)
                    if e1 > 1e-2:
                        success = False
                        
                if v_norm is not None:
                    V[:, c] = v_norm
                else:
                    V[:, c], e2 = _project_half_integer(v)
                    if e2 > 1e-2:
                        success = False
                        
                if w_norm is not None:
                    W[:, c] = w_norm
                else:
                    W[:, c], e3 = _project_half_integer(w)
                    if e3 > 1e-2:
                        success = False

            if success:
                # Verify with the projected (half-integer) values
                if _verify(U, V, W, n, m, p):
                    # Found a valid decomposition of rank r
                    result = {
                        "U": _to_list(U),
                        "V": _to_list(V),
                        "W": _to_list(W),
                    }
                    return result
        except Exception:
            continue

    return best


# EVOLVE-BLOCK-END
