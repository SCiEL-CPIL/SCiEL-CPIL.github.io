"""Suite task: cover unit sphere with 130 equal caps (minimize covering radius)."""
import numpy as np

N = 130
TIME_LIMIT = 112.0  # safety margin under 120s wall-clock


def _facet_center(a, b, c):
    """Compute the outward-pointing circumcenter direction of a spherical triangle."""
    for e in ((a, b, c), (b, a, c), (a, c, b)):
        cand = np.cross(e[0], e[1])
        if np.dot(cand, e[2]) > 0 and np.linalg.norm(cand) > 1e-12:
            ctr = cand / np.linalg.norm(cand)
            if np.dot(ctr, (a + b + c)) < 0:
                ctr = -ctr
            return ctr
    ctr = a + b + c
    n = np.linalg.norm(ctr)
    if n < 1e-9:
        return None
    return ctr / n


def _cover_radius(p):
    """Exact covering radius in radians = max circumradius of hull facets (empty caps)."""
    from scipy.spatial import ConvexHull
    p = p - p.mean(axis=0, keepdims=True)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    if np.any(~np.isfinite(p)):
        return np.inf
    try:
        h = ConvexHull(p)
    except Exception:
        return np.inf
    max_r = 0.0
    for f in h.simplices:
        a, b, c = p[f[0]], p[f[1]], p[f[2]]
        ctr = _facet_center(a, b, c)
        if ctr is None:
            continue
        r = 0.0
        for v in (a, b, c):
            r = max(r, np.arccos(np.clip(np.dot(ctr, v), -1, 1)))
        max_r = max(max_r, r)
    return max_r


def _random_seed(rng, noise_scale=0.03):
    """Generate a perturbed golden-angle spiral as a warm start."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    P = P + noise_scale * rng.standard_normal((N, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P


def _random_few_points(rng, n_few=20):
    """Generate a fully random configuration with a few fixed anchor points."""
    P = rng.standard_normal((N, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    # Add some structured points to help convergence
    i = np.arange(n_few)
    z = 1.0 - 2.0 * (i + 0.5) / n_few
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    anchors = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    P[:n_few] = anchors
    return P


def _steepest_descent(P, time_budget):
    """Iteratively push the hot facet's vertices apart to shrink the max empty cap."""
    import time
    from scipy.spatial import ConvexHull
    t0 = time.time()
    P = P.copy()
    P -= P.mean(axis=0, keepdims=True)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    best = _cover_radius(P)
    bestP = P.copy()
    step = 0.005
    mom = np.zeros_like(P)
    no_improve = 0
    while time.time() - t0 < time_budget:
        try:
            h = ConvexHull(P)
        except Exception:
            P = bestP.copy()
            break
        # find hot facet
        worst = -1.0
        hot = None
        hot_ctr = None
        for f in h.simplices:
            a, b, c = P[f[0]], P[f[1]], P[f[2]]
            ctr = _facet_center(a, b, c)
            if ctr is None:
                continue
            r = max(np.arccos(np.clip(np.dot(ctr, v), -1, 1)) for v in (a, b, c))
            if r > worst:
                worst = r
                hot = f
                hot_ctr = ctr
        if hot is None:
            break
        # steepest descent: move each hot vertex away from the facet center
        g = np.zeros_like(P)
        for idx in hot:
            g[idx] -= hot_ctr
        gm = np.linalg.norm(g, axis=1, keepdims=True)
        gm = np.where(gm > 0, gm, 1.0)
        g = g / gm
        mom = 0.6 * mom + 0.4 * g
        Pnew = P - step * mom
        Pnew -= Pnew.mean(axis=0, keepdims=True)
        Pnew /= np.linalg.norm(Pnew, axis=1, keepdims=True)
        val = _cover_radius(Pnew)
        if val < best - 1e-13:
            P = Pnew
            best = val
            bestP = Pnew.copy()
            no_improve = 0
            step = min(step * 1.05, 0.02)  # allow slightly larger steps when improving
        else:
            no_improve += 1
            step *= 0.8
            if step < 1e-8:
                step = 0.005
                mom *= 0.1
            if no_improve > 200:
                break
    return bestP, best


def construct():
    import time
    t_start = time.time()
    rng = np.random.default_rng(12345)
    bestP = None
    best = np.inf

    # initial golden-angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    bestP = P0.copy()
    best = _cover_radius(P0)

    # Polish the initial spiral first
    remaining = TIME_LIMIT - (time.time() - t_start)
    if remaining > 15:
        budget = min(25.0, remaining - 10)
        Pc, vc = _steepest_descent(P0, budget)
        if vc < best:
            best = vc
            bestP = Pc.copy()

    # Several diverse warm-starts
    n_starts = 12
    for s in range(n_starts):
        elapsed = time.time() - t_start
        if elapsed > TIME_LIMIT - 30:
            break
        if s % 3 == 0:
            Pseed = _random_seed(rng, noise_scale=0.02 + 0.01 * s)
        elif s % 3 == 1:
            Pseed = _random_few_points(rng, n_few=15 + 5 * (s % 4))
        else:
            Pseed = _random_seed(rng, noise_scale=0.05)
        remaining = TIME_LIMIT - (time.time() - t_start)
        if remaining < 12:
            break
        budget = min(18.0, max(8.0, remaining - 15))
        Pc, vc = _steepest_descent(Pseed, budget)
        if vc < best:
            best = vc
            bestP = Pc.copy()

    # Final local polish on the best configuration if time remains
    elapsed = time.time() - t_start
    if elapsed < TIME_LIMIT - 10:
        budget = TIME_LIMIT - 12 - elapsed
        if budget > 3:
            Pc, vc = _steepest_descent(bestP, budget)
            if vc < best:
                best = vc
                bestP = Pc.copy()

    # safety: ensure valid, finite, on-sphere
    bestP = np.asarray(bestP, dtype=float)
    bestP -= bestP.mean(axis=0, keepdims=True)
    bestP /= np.linalg.norm(bestP, axis=1, keepdims=True)
    return bestP
