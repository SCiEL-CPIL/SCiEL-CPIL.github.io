# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 600.0  # safety margin below the 630s wall-clock kill


def _compute_P(support, weights):
    """Exact P[X1+X2+X3 < 2*X4] for i.i.d. draws from the discrete measure.

    Uses sorted pair-sums + searchsorted for O(n^2 log n) cost instead of O(n^4).
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    # All pairwise sums S_ij = s_i + s_j with joint weight w_i*w_j, sorted.
    S = (s[:, None] + s[None, :]).ravel()
    W = (w[:, None] * w[None, :]).ravel()
    order = np.argsort(S, kind="stable")
    Ss = S[order]
    Ws = W[order]
    cdf = np.cumsum(Ws)
    total = cdf[-1]
    # For each (k, l) pair, count pair-sums strictly less than 2*s_l - s_k.
    targets = 2.0 * s[None, :] - s[:, None]  # shape (n, n), [k, l]
    # searchsorted 'right' gives count of elements <= target; we need strictly <.
    # count of Ss < t == searchsorted(Ss, t, side='left').
    idx = np.searchsorted(Ss, targets.ravel(), side="left")  # shape (n*n,)
    # cdf[idx] is sum of Ws for elements with index < idx, i.e. Ss < target.
    # Guard idx==0 -> cdf[0] would be wrong; use where.
    inner = np.where(idx > 0, cdf[np.maximum(idx - 1, 0)], 0.0)
    inner = inner.reshape(n, n)  # [k, l]
    # P = sum_{k,l} w_k w_l * inner_{k,l} / total
    P = (w[:, None] * w[None, :] * inner).sum() / total
    return float(P)


def _validate(support, weights):
    """Return True if the (support, weights) pair is a valid discrete measure."""
    if not isinstance(support, np.ndarray) or not isinstance(weights, np.ndarray):
        return False
    if support.ndim != 1 or weights.ndim != 1:
        return False
    n = support.size
    if n < 1 or n != weights.size:
        return False
    if not np.all(np.isfinite(support)) or not np.all(np.isfinite(weights)):
        return False
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return False
    if weights.sum() <= 1e-9:
        return False
    return True


def _two_point_P(q, a, b):
    """P for a two-point measure: mass (1-q) at a, mass q at b.

    Inequality a+a+a < 2b holds iff 3a < 2b.
    If 3a < 2b: P = 1 - q^4 (holds unless all three of X1,X2,X3 are b).
    If 3a >= 2b: P = (1-q)^3 * q  (only holds when X4=b and X1=X2=X3=a).
    """
    if 3.0 * a < 2.0 * b:
        return 1.0 - q ** 4
    else:
        return (1.0 - q) ** 3 * q


def _refine_two_point(a, b, q, time_budget):
    """Nelder-Mead refine a two-point measure (a, b, q). Returns best (a,b,q,P)."""
    t0 = time.time()
    best = (a, b, q, _two_point_P(q, a, b))

    def obj(x):
        a_, b_, q_ = x
        if a_ < 0 or b_ < a_ or q_ <= 0 or q_ >= 1:
            return 1e6
        return -_two_point_P(q_, a_, b_)

    # Build an initial simplex around the starting point.
    a0, b0, q0 = a, b, q
    simplex = [
        [a0, b0, q0],
        [a0 * 1.05 + 1e-3, b0 * 1.05, q0 * 0.9],
        [a0 * 0.95, b0 * 1.1, q0 * 1.1],
    ]
    # Clamp q into (0,1).
    simplex = [[max(1e-6, min(1 - 1e-6, x[2])), x[1], max(1e-6, min(1 - 1e-6, x[2]))] for x in simplex]
    # Rebuild with proper clamping on all coords.
    def clamp(x):
        a_ = max(0.0, x[0])
        b_ = max(a_ + 1e-6, x[1])
        q_ = max(1e-6, min(1 - 1e-6, x[2]))
        return [a_, b_, q_]
    simplex = [clamp(x) for x in simplex]

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    fvals = [obj(x) for x in simplex]
    best_x = simplex[int(np.argmin(fvals))]
    best_f = min(fvals)

    for _ in range(4000):
        if time.time() - t0 > time_budget:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if fvals[0] < best_f:
            best_f = fvals[0]
            best_x = simplex[0]
        worst = simplex[-1]
        if abs(fvals[-1] - fvals[0]) < 1e-12:
            break
        # Centroid of all but worst.
        centroid = np.mean(simplex[:-1], axis=0)
        # Reflection.
        xr = clamp(centroid + alpha * (centroid - np.array(worst)))
        fr = obj(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        elif fr < fvals[0]:
            # Expansion.
            xe = clamp(centroid + gamma * (centroid - np.array(worst)))
            fe = obj(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        else:
            # Contraction.
            xc = clamp(centroid + rho * (np.array(worst) - centroid))
            fc = obj(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink.
                for i in range(1, len(simplex)):
                    simplex[i] = clamp(simplex[0] + sigma * (np.array(simplex[i]) - np.array(simplex[0])))
                    fvals[i] = obj(simplex[i])
    a_, b_, q_ = best_x
    return a_, b_, q_, _two_point_P(q_, a_, b_)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """Construct a discrete measure maximizing P[X1+X2+X3 < 2*X4]."""
    t_start = time.time()
    remaining = lambda: _DEADLINE - (time.time() - t_start)

    # Phase 1: diverse two-point seeds, refine each with a small budget.
    # A two-point measure with tiny mass at a large b gives P = 1 - q^4 ~ 1.
    seeds = []
    # Base strong seed: a=0, b large, q tiny.
    seeds.append((0.0, 50.0, 0.001))
    # Vary b and q.
    for b in (20.0, 40.0, 80.0, 120.0, 200.0):
        for q in (0.0005, 0.002, 0.01, 0.05):
            seeds.append((0.0, b, q))
    # Some with a > 0.
    for a in (1.0, 2.0, 5.0):
        for b in (30.0, 60.0, 100.0):
            for q in (0.001, 0.005, 0.02):
                seeds.append((a, b, q))

    best_P = -1.0
    best_two = None
    per_seed_budget = 6.0  # seconds per seed refinement
    for (a, b, q) in seeds:
        if remaining() < 30.0:
            break
        a_r, b_r, q_r, P_r = _refine_two_point(a, b, q, per_seed_budget)
        if P_r > best_P:
            best_P = P_r
            best_two = (a_r, b_r, q_r)

    # Build the two-point measure from the best refined seed.
    if best_two is not None:
        a, b, q = best_two
        support = np.array([a, b], dtype=np.float64)
        weights = np.array([1.0 - q, q], dtype=np.float64)
        # Normalize.
        weights = weights / weights.sum()
        P_best = _compute_P(support, weights)
    else:
        # Fallback to uniform grid.
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        P_best = _compute_P(support, weights)

    # Phase 2: try to improve with a small multi-point measure via Nelder-Mead
    # over a compact parameterization, but only if we have ample time left.
    if remaining() > 120.0:
        # Try a 3-point measure: mass at a, b, c with b large.
        # Parameterize by (a, b, c, p, q) where weights are (1-p-q, p, q).
        def make_3pt(a, b, c, p, q):
            s = np.array([a, b, c], dtype=np.float64)
            w = np.array([1.0 - p - q, p, q], dtype=np.float64)
            if np.any(w < 0) or np.any(s < 0):
                return None, None, -1.0
            w = np.maximum(w, 0.0)
            if w.sum() <= 1e-9:
                return None, None, -1.0
            w = w / w.sum()
            P = _compute_P(s, w)
            return s, w, P

        # Seed from best two-point: split the large mass into two close points.
        if best_two is not None:
            a, b, q = best_two
            # 3-point seeds: keep a, split b into b1, b2.
            three_seeds = [
                (a, b * 0.9, b * 1.1, 0.0, q),
                (a, b, b * 1.2, 0.0, q),
                (a, b * 0.95, b * 1.05, 0.0, q),
            ]
        else:
            three_seeds = [(0.0, 40.0, 50.0, 0.0, 0.01)]

        for (a, b, c, p, q) in three_seeds:
            if remaining() < 60.0:
                break
            s, w, P = make_3pt(a, b, c, p, q)
            if P > P_best and s is not None:
                P_best = P
                support, weights = s, w

    # Final validation and return.
    if not _validate(support, weights):
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
    return support, weights


# EVOLVE-BLOCK-END
