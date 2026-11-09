import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110.0  # safety margin before the 150s kill

    # Schoolbook (always valid) as the guaranteed fallback.
    def _schoolbook():
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
        return (U, V, W)

    def _verify(U, V, W):
        # Exact integer check: tensor_{i,j,k} = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # must equal 1 for every (i,j,k) and 0 otherwise.
        Ua = np.asarray(U)
        Va = np.asarray(V)
        Wa = np.asarray(W)
        # For each (i,j,k): product of the three scalars, summed over r.
        # Build via einsum-free explicit loops (small n,m,p).
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(Ua.shape[1]):
                        s += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                    if s != 1:
                        return False
        # Check a few off-diagonal (i,j,k) combos must be 0.
        # (i,j) and (j,k) and (k,i) must be mutually consistent with a single j.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # off-diagonal: pick (i', j', k') that is not a valid triple
                    pass
        # Stronger: verify the full tensor equals the multiplication tensor by
        # reconstructing it and comparing against the known 1/0 pattern.
        T = np.zeros((n * m, m * p, p * n))
        for r in range(Ua.shape[1]):
            T += Ua[:, r][:, None, None] * Va[:, r][None, :, None] * Wa[:, r][None, None, :]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[i * m + j, j * p + k, k * n + i] != 1:
                        return False
        # all other entries must be 0
        mask = np.ones(T.shape, dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    mask[i * m + j, j * p + k, k * n + i] = False
        if np.any(T[mask] != 0):
            return False
        return True

    def _build(n, m, p):
        """Recursive bilinear construction. Returns (U,V,W) as np arrays, all-integer.

        Index conventions:
          U row = i*m + j   (i in [0,n), j in [0,m))
          V row = j*p + k   (j in [0,m), k in [0,p))
          W row = k*n + i   (k in [0,p), i in [0,n))
        """
        def _base(n, m, p):
            # schoolbook for this block
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
            return (U, V, W)

        def _hstack3(U1, U2, U3):
            # concatenate along columns, rows identical shape
            return np.hstack([U1, U2, U3])

        # Split m into m1 + m2 (m2 = 1).
        m1 = m - 1
        m2 = 1
        # Split p into p1 + p2 (balanced).
        p1 = p // 2
        p2 = p - p1

        # U1 (n x m1), V1 (m1 x p1)
        U1, V1, W1 = _build(n, m1, p1)
        # U1 (n x m1), V2 (m1 x p2)
        U1b, V2, W2 = _build(n, m1, p2)
        # U3 (n x 1), V3 (1 x p)
        U3, V3, W3 = _build(n, 1, p)

        # Combine: C = A1 B1 + A1 B2 + A3 B3
        # U = [U1, U1b, U3] (same n x m rows), V = [V1, V2, V3], W = [W1, W2, W3]
        U = np.hstack([U1, U1b, U3])
        V = np.hstack([V1, V2, V3])
        W = np.hstack([W1, W2, W3])
        return (U, V, W)

    try:
        U, V, W = _build(n, m, p)
        if time.time() < deadline - 5.0 and _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook (always valid).
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
