import numpy as np
import time

DIMENSIONS = [3, 3, 6]
START = time.time()
DEADLINE = 150.0 - 10.0


def _ok():
    return time.time() < DEADLINE


def _verify_3x3(u, v, w, iters=12, seed=999):
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        a = rng.integers(-5, 6, size=(3, 3))
        b = rng.integers(-5, 6, size=(3, 3))
        c_true = a @ b
        c = np.zeros((3, 3))
        for t in range(u.shape[0]):
            alpha = float(np.sum(u[t] * a))
            beta = float(np.sum(v[t] * b))
            c += alpha * beta * w[t]
        if not np.allclose(c, c_true, atol=1e-9, rtol=0.0):
            return False
    return True


def _find_3x3(rank, seed, trials=400):
    rng = np.random.default_rng(seed)
    # Use a broader value set for better chances
    vals = np.array([-2, -1, 0, 1, 2, 3])
    for trial in range(trials):
        if not _ok():
            return None
        # Mix different value ranges for diversity
        if trial % 3 == 0:
            u = rng.choice(vals, size=(rank, 3, 3))
            v = rng.choice(vals, size=(rank, 3, 3))
        elif trial % 3 == 1:
            u = rng.choice(np.array([-1, 0, 1]), size=(rank, 3, 3))
            v = rng.choice(vals, size=(rank, 3, 3))
        else:
            u = rng.choice(vals, size=(rank, 3, 3))
            v = rng.choice(np.array([-1, 0, 1]), size=(rank, 3, 3))
        
        rows = []
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    row = np.zeros(rank * 9)
                    for t in range(rank):
                        row[t * 9 + i * 3 + j] = u[t, i, k] * v[t, k, j]
                    rows.append(row)
        A = np.array(rows)
        try:
            w, *_ = np.linalg.lstsq(A, np.ones(27), rcond=None)
        except np.linalg.LinAlgError:
            continue
        w = w.reshape(rank, 3, 3)
        wr = np.round(w * 2) / 2  # round to nearest half-integer
        if _verify_3x3(u, v, wr):
            return u.astype(float), v.astype(float), wr
    return None


def _build_3x6(r, d0, d1):
    n, m, p = 3, 3, 6
    R = 2 * r
    U = np.zeros((9, R))
    V = np.zeros((18, R))
    W = np.zeros((18, R))
    for q, (u, v, w) in enumerate((d0, d1)):
        base = q * r
        for t in range(r):
            col = base + t
            for i in range(3):
                for k in range(3):
                    U[i * 3 + k, col] = u[t, i, k]
            for k in range(3):
                for j in range(3):
                    V[k * 6 + (q * 3 + j), col] = v[t, k, j]
            for i in range(3):
                for j in range(3):
                    W[(q * 3 + j) * 3 + i, col] = w[t, i, j]
    return U, V, W


def _verify_3x6(U, V, W, iters=12, seed=555):
    n, m, p = 3, 3, 6
    R = U.shape[1]
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        a = rng.integers(-4, 5, size=(n, m))
        b = rng.integers(-4, 5, size=(m, p))
        C = a @ b
        Cc = np.zeros((n, p))
        for c in range(R):
            ua = np.array([[U[i * m + k, c] for k in range(m)] for i in range(n)])
            vb = np.array([[V[k * p + j, c] for j in range(p)] for k in range(m)])
            wa = np.array([[W[j * n + i, c] for i in range(n)] for j in range(p)])
            Cc += float(np.sum(ua * a)) * float(np.sum(vb * b)) * wa
        if not np.allclose(Cc, C, atol=1e-9, rtol=0.0):
            return False
    return True


def _find_3x6_direct(rank, seed, trials=200):
    """Direct search for 3x3x6 decomposition"""
    n, m, p = 3, 3, 6
    rng = np.random.default_rng(seed)
    vals = np.array([-2, -1, 0, 1, 2])
    for _ in range(trials):
        if not _ok():
            return None
        U = rng.choice(vals, size=(rank, n * m))
        V = rng.choice(vals, size=(rank, m * p))
        
        # Build constraint matrix: for each (i,j), sum_t U[t, i*m+j] * V[t, k*p+l] * W[t, l*n+i] = delta
        # We need: C[i,l] = sum_k A[i,k] * B[k,l]
        # C[i,l] = sum_t (sum_k U[t,i,k] * A[i,k]) * (sum_k V[t,k,l] * B[k,l]) * W[t,l,i]
        # This must equal sum_k A[i,k] * B[k,l]
        
        # The constraint is: for all i,l: sum_t U[t,i,·]·A[i,·] * V[t,·,l]·B[·,l] * W[t,l,i] = sum_k A[i,k]*B[k,l]
        # This is a trilinear form. We can set up linear equations in W.
        
        # For each (i, l) pair, we need:
        # sum_t W[t, l*n+i] * (sum_k U[t,i,k]*A[i,k]) * (sum_k V[t,k,l]*B[k,l]) = sum_k A[i,k]*B[k,l]
        
        # But this depends on A and B, so we need it to hold for all A,B.
        # The tensor equation is: T_{i,k,l} = sum_t U[t,i,k] * V[t,k,l] * W[t,l,i]
        # where T_{i,k,l} = 1 if we're computing C[i,l] from A[i,k]*B[k,l]
        
        # Actually, the multiplication tensor has T[i,k,l] = 1 for the (i,k,l) term
        # and we need: sum_t U[t,i,k] * V[t,k,l] * W[t,l,i] = 1 for all i,k,l
        
        rows = []
        rhs = []
        for i in range(n):
            for k in range(m):
                for l in range(p):
                    row = np.zeros(rank)
                    for t in range(rank):
                        row[t] = U[t, i * m + k] * V[t, k * p + l]
                    rows.append(row)
                    rhs.append(1.0)
        
        A_mat = np.array(rows)
        b_vec = np.array(rhs)
        
        try:
            W_flat, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        W = W_flat.reshape(rank, p * n)
        Wr = np.round(W * 2) / 2
        
        # Verify
        if _verify_3x6(U.astype(float), V.astype(float), Wr):
            return U.astype(float), V.astype(float), Wr
    return None


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    # Try direct 3x3x6 search for ranks below 54
    for r in range(40, 54):
        if not _ok():
            break
        result = _find_3x6_direct(r, seed=5000 + r, trials=100)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to two 3x3x3 blocks
    for r in range(21, 28):
        if not _ok():
            break
        d0 = _find_3x3(r, seed=1000 + r, trials=300)
        if d0 is None:
            continue
        d1 = _find_3x3(r, seed=3000 + r, trials=300)
        if d1 is None:
            continue
        U, V, W = _build_3x6(r, d0, d1)
        if _verify_3x6(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return _schoolbook()
