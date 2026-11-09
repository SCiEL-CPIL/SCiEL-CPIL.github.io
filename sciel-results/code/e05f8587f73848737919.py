# EVOLVE-BLOCK-START
import time
import numpy as np


def _shannon(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[v > 1e-15]
    if v.size == 0:
        return 0.0
    return float(-(v * np.log2(v)).sum())


def _compute_ratio(P):
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        return 0.0
    P = P / s
    n = P.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    xx = np.tile(x, (n, 1))
    yy = np.tile(y, (n, 1)).T

    # Diagonal (x - y) pooled distribution
    diag = np.bincount(xx - yy + (n - 1), weights=P.ravel(), minlength=2 * n - 1)
    D = _shannon(diag)

    # Marginals
    p_x = P.sum(axis=0)
    H0 = _shannon(p_x)
    p_xy = np.bincount(xx + yy, weights=P.ravel(), minlength=2 * n - 1)
    H1 = _shannon(p_xy)
    p_2y = np.bincount(xx + 2 * yy, weights=P.ravel(), minlength=3 * n - 1)
    H2 = _shannon(p_2y)
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M


def _structured_candidates(n=10):
    cands = []
    # Uniform baseline
    cands.append(np.ones((n, n)) / (n * n))
    # Pure diagonals (maximize H_{-1})
    for d in range(2 * n - 1):
        A = np.zeros((n, n))
        for y in range(n):
            x = y + (d - (n - 1))
            if 0 <= x < n:
                A[y, x] = 1.0
        cands.append(A / A.sum())
    # Pure columns (x fixed) -> H0=0, H1,H2 spread
    for x in range(n):
        A = np.zeros((n, n)); A[:, x] = 1.0
        cands.append(A / n)
    # Pure rows (y fixed) -> H1=H2=0
    for y in range(n):
        A = np.zeros((n, n)); A[y, :] = 1.0
        cands.append(A / n)
    # Concentrate on a few diagonals
    for k in range(1, n):
        for d0 in range(0, 2 * n - 1 - k):
            A = np.zeros((n, n))
            for d in range(d0, d0 + k):
                for y in range(n):
                    x = y + (d - (n - 1))
                    if 0 <= x < n:
                        A[y, x] = 1.0
            if A.sum() > 0:
                cands.append(A / A.sum())
    # Mixtures of two diagonals
    for d1 in range(2 * n - 1):
        for d2 in range(d1 + 1, 2 * n - 1):
            A1 = np.zeros((n, n)); A2 = np.zeros((n, n))
            for y in range(n):
                x1 = y + (d1 - (n - 1))
                if 0 <= x1 < n:
                    A1[y, x1] = 1.0
                x2 = y + (d2 - (n - 1))
                if 0 <= x2 < n:
                    A2[y, x2] = 1.0
            if A1.sum() > 0 and A2.sum() > 0:
                for w in (0.25, 0.4, 0.5, 0.6, 0.75):
                    A = w * A1 + (1 - w) * A2
                    cands.append(A / A.sum())
    # Random sparse supports
    rng = np.random.default_rng(12345)
    for _ in range(200):
        A = np.zeros((n, n))
        m = rng.integers(4, 25)
        idx = rng.choice(n * n, size=m, replace=False)
        A.flat[idx] = 1.0
        cands.append(A / A.sum())
    return cands


def _reweight(P, n=10, iters=2000, alpha=0.3, beta=0.25, eta=0.1):
    """Multiplicative-weights style update: push mass to high-diagonal cells,
    pull mass away from cells that inflate H0/H1/H2."""
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        return P
    P = P / s
    x = np.arange(n)
    y = np.arange(n)
    xx = np.tile(x, (n, 1))
    yy = np.tile(y, (n, 1)).T
    diag = xx - yy + (n - 1)
    # Precompute per-cell contribution to each marginal bin
    bin0 = xx
    bin1 = xx + yy
    bin2 = xx + 2 * yy
    best = P.copy()
    best_score = _compute_ratio(best)
    for it in range(iters):
        # Pool probs
        p_diag = np.bincount(diag.ravel(), weights=P.ravel(), minlength=2 * n - 1)
        p_0 = np.bincount(bin0.ravel(), weights=P.ravel(), minlength=n)
        p_1 = np.bincount(bin1.ravel(), weights=P.ravel(), minlength=2 * n - 1)
        p_2 = np.bincount(bin2.ravel(), weights=P.ravel(), minlength=3 * n - 1)
        # Gradient of D w.r.t. P: dD/dP_ij = -log2(p_diag) - 1/ln2  (for p_diag>0)
        def grad_entropy(p):
            g = np.zeros_like(p)
            m = p > 1e-15
            g[m] = -np.log2(p[m]) - 1.0 / np.log(2.0)
            return g
        gD = grad_entropy(p_diag)
        g0 = grad_entropy(p_0)
        g1 = grad_entropy(p_1)
        g2 = grad_entropy(p_2)
        # cell gradient: alpha * gD[diag] - beta * max(g0[bin0], g1[bin1], g2[bin2])
        cell_g = (alpha * gD[diag]
                  - beta * np.maximum(np.maximum(g0[bin0], g1[bin1]), g2[bin2]))
        # multiplicative update
        P = P * np.exp(eta * cell_g)
        P = np.clip(P, 0.0, None)
        s = P.sum()
        if s <= 1e-9:
            break
        P = P / s
        sc = _compute_ratio(P)
        if sc > best_score:
            best_score = sc
            best = P.copy()
    return best


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start = time.time()
    deadline = start + 600.0
    margin = 15.0

    cands = _structured_candidates(n)
    best = cands[0].copy()
    best_score = _compute_ratio(best)

    # Evaluate all structured candidates
    for A in cands[1:]:
        if time.time() > deadline - margin:
            break
        sc = _compute_ratio(A)
        if sc > best_score:
            best_score = sc
            best = A.copy()

    # Reweighting from the best structured candidate
    P = _reweight(best, n=n, iters=1500, alpha=0.3, beta=0.25, eta=0.1)
    sc = _compute_ratio(P)
    if sc > best_score:
        best_score = sc
        best = P.copy()

    # Additional reweighting passes with different hyperparameters if time allows
    if time.time() < deadline - margin:
        P2 = _reweight(best, n=n, iters=1500, alpha=0.5, beta=0.2, eta=0.08)
        sc2 = _compute_ratio(P2)
        if sc2 > best_score:
            best_score = sc2
            best = P2.copy()

    if time.time() < deadline - margin:
        P3 = _reweight(best, n=n, iters=1500, alpha=0.2, beta=0.35, eta=0.12)
        sc3 = _compute_ratio(P3)
        if sc3 > best_score:
            best_score = sc3
            best = P3.copy()

    # Final coordinate-descent polish: try small perturbations
    if time.time() < deadline - margin:
        Pp = best.copy()
        for _ in range(200):
            if time.time() > deadline - margin:
                break
            y, x = np.random.randint(n), np.random.randint(n)
            y2, x2 = np.random.randint(n), np.random.randint(n)
            eps = 1e-3
            Pp[y, x] += eps
            Pp[y2, x2] -= eps
            Pp = np.clip(Pp, 0.0, None)
            s = Pp.sum()
            if s <= 1e-9:
                Pp = best.copy(); continue
            Pp = Pp / s
            scp = _compute_ratio(Pp)
            if scp > best_score:
                best_score = scp
                best = Pp.copy()
            else:
                Pp = best.copy()

    # Normalize and return
    best = np.clip(best, 0.0, None)
    s = best.sum()
    if s <= 1e-9:
        return np.ones((n, n)) / (n * n)
    return best / s


# EVOLVE-BLOCK-END
