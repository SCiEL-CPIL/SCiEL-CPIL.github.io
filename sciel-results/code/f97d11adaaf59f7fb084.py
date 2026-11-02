"""Matrix multiplication 2x3x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 105.0  # safety margin before 120s

    n, m, p = DIMENSIONS
    dimU, dimV, dimW = n * m, m * p, p * n  # 6, 18, 12

    # Build multiplication tensor T[a,b,c]
    T = np.zeros((dimU, dimV, dimW), dtype=np.float64)
    for a in range(dimU):
        i, j = divmod(a, m)
        for b in range(dimV):
            j2, k = divmod(b, p)
            if j != j2:
                continue
            for c in range(dimW):
                k2, i2 = divmod(c, n)
                if k == k2 and i == i2:
                    T[a, b, c] = 1.0

    T1 = T.reshape(dimU, dimV * dimW)  # (6, 216)
    T2 = T.reshape(dimV, dimU * dimW)  # (18, 72)
    T3 = T.reshape(dimW, dimU * dimV)  # (12, 108)

    def to_half(x):
        return np.round(2.0 * x) / 2.0

    def is_half(M, tol=1e-7):
        return bool(np.all(np.abs(2.0 * M - np.round(2.0 * M)) < tol))

    def reconstruct(U, V, W):
        return np.einsum('ic,jc,kc->ijk', U, V, W)

    def residual(U, V, W):
        return float(np.linalg.norm(reconstruct(U, V, W) - T))

    def verify(U, V, W, tol=1e-8):
        if not (is_half(U) and is_half(V) and is_half(W)):
            return False
        return residual(U, V, W) < tol

    def hosvd_init(r, seed):
        rng = np.random.default_rng(seed)
        # SVD of unfoldings, take top-r left singular vectors
        U = np.linalg.svd(T1, full_matrices=False)[0][:, :r]
        V = np.linalg.svd(T2, full_matrices=False)[0][:, :r]
        W = np.linalg.svd(T3, full_matrices=False)[0][:, :r]
        # random sign/phase to help
        return U, V, W

    def als(r, n_iter=120, seed=0, init='hosvd'):
        if init == 'hosvd':
            U, V, W = hosvd_init(r, seed)
        else:
            rng = np.random.default_rng(seed)
            U = rng.standard_normal((dimU, r))
            V = rng.standard_normal((dimV, r))
            W = rng.standard_normal((dimW, r))
        prev = np.inf
        for it in range(n_iter):
            if time.time() > deadline - 3:
                break
            VW = np.einsum('jc,kc->jkc', V, W).reshape(dimV * dimW, r)
            U = np.linalg.lstsq(VW, T1.T, rcond=None)[0].T
            UW = np.einsum('ic,kc->ikc', U, W).reshape(dimU * dimW, r)
            V = np.linalg.lstsq(UW, T2.T, rcond=None)[0].T
            UV = np.einsum('ic,jc->ijc', U, V).reshape(dimU * dimV, r)
            W = np.linalg.lstsq(UV, T3.T, rcond=None)[0].T
            res = residual(U, V, W)
            if abs(prev - res) < 1e-12 and it > 10:
                break
            prev = res
        return U, V, W

    # Schoolbook fallback (always valid, rank 36)
    def schoolbook():
        r = n * m * p
        U = np.zeros((dimU, r)); V = np.zeros((dimV, r)); W = np.zeros((dimW, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    best = None  # (rank, U, V, W)
    # Lower bound from W-flattening rank = 12
    for r in range(12, n * m * p + 1):
        if time.time() > deadline - 8:
            break
        found = False
        n_restart = 6 if r < 25 else 4
        for s in range(n_restart):
            if time.time() > deadline - 8:
                break
            init = 'hosvd' if s % 2 == 0 else 'rand'
            U, V, W = als(r, n_iter=120, seed=1000 + s, init=init)
            # polish: a few more iters
            for _ in range(20):
                VW = np.einsum('jc,kc->jkc', V, W).reshape(dimV * dimW, r)
                U = np.linalg.lstsq(VW, T1.T, rcond=None)[0].T
                UW = np.einsum('ic,kc->ikc', U, W).reshape(dimU * dimW, r)
                V = np.linalg.lstsq(UW, T2.T, rcond=None)[0].T
                UV = np.einsum('ic,jc->ijc', U, V).reshape(dimU * dimV, r)
                W = np.linalg.lstsq(UV, T3.T, rcond=None)[0].T
            if residual(U, V, W) < 1e-6:
                Uh, Vh, Wh = to_half(U), to_half(V), to_half(W)
                if verify(Uh, Vh, Wh):
                    best = (r, Uh, Vh, Wh)
                    found = True
                    break
        if found:
            break

    if best is None:
        U, V, W = schoolbook()
    else:
        U, V, W = best[1], best[2], best[3]

    # Convert to lists
    U = U.tolist(); V = V.tolist(); W = W.tolist()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
