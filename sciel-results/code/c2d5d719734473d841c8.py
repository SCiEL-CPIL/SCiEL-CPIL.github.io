"""Pack 200 equal circles in a unit square via hex-grid greedy + local polish."""
import numpy as np
import time

N = 200
HALF = 0.5


def _score(c):
    """Return the common radius for a configuration (higher is better)."""
    c = np.asarray(c)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # inside square
    m = np.max(np.abs(c), axis=1)
    if np.any(m > HALF + 1e-12):
        return 0.0
    # pairwise min distance
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    return min(min_pair / 2.0, HALF - np.max(m))


def _hex_points(s, ox, oy):
    """Hexagonal lattice points with spacing s, offset (ox, oy)."""
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    cols = int(np.ceil(1.0 / dx)) + 3
    rows = int(np.ceil(1.0 / dy)) + 3
    xs = np.arange(-2, cols) * dx + ox
    ys = np.arange(-2, rows) * dy + oy
    # offset every other column
    pts = []
    for j, x in enumerate(xs):
        yoff = (dy / 2.0) if (j % 2 == 1) else 0.0
        for i, y in enumerate(ys):
            pts.append((x + 0.0, y + yoff))
    arr = np.array(pts)
    # keep inside square
    keep = np.all(np.abs(arr) <= HALF, axis=1)
    return arr[keep]


def _greedy_select(cands, k, r_min):
    """Greedy farthest-point selection of k points maximizing min distance."""
    n = cands.shape[0]
    if n <= k:
        return cands.copy()
    selected = []
    # start from a random candidate
    idx = int(np.random.randint(n))
    selected.append(idx)
    min_dist = np.full(n, np.inf)
    for _ in range(k - 1):
        # update min distance to selected
        last = cands[selected[-1]]
        d = np.linalg.norm(cands - last, axis=1)
        min_dist = np.minimum(min_dist, d)
        # pick the farthest
        j = int(np.argmax(min_dist))
        if min_dist[j] < r_min - 1e-9:
            break
        selected.append(j)
    return cands[selected]


def _polish(c, iters=600, lr0=1e-3):
    """Local gradient-free polish: maximize min of (pairwise dist/2, margin)."""
    c = c.copy()
    best = _score(c)
    # use a smooth-ish surrogate: penalize violations
    def surrogate(x):
        d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        np.fill_diagonal(d, 1e9)
        pd = d / 2.0
        margin = HALF - np.max(np.abs(x), axis=1)
        # soft min via negative log-sum-exp of negatives
        # we want min of all constraints; use min of (pd min, margin min)
        return min(np.min(pd), np.min(margin))
    # simple coordinate ascent with adaptive step
    rng = np.random.default_rng(12345)
    step = lr0
    for it in range(iters):
        i = int(rng.integers(N))
        d = int(rng.integers(2))
        # try a few step sizes
        improved = False
        for sgn in (1.0, -1.0):
            for sc in (step, step * 0.5, step * 2.0):
                c2 = c.copy()
                c2[i, d] += sgn * sc
                if not np.all(np.abs(c2) <= HALF):
                    continue
                v = _score(c2)
                if v > best + 1e-12:
                    c = c2
                    best = v
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step *= 0.7
        else:
            step = min(step * 1.1, 5e-3)
    return c, best


def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s

    rng = np.random.default_rng(2024)

    best_c = None
    best_v = 0.0

    # target spacing estimate
    s0 = 1.0 / np.sqrt(N * np.sqrt(3.0) / 2.0)  # ~0.076

    # multiple hex-grid trials with random offsets and varied spacing
    n_trials = 40
    for trial in range(n_trials):
        if time.time() > DEADLINE:
            break
        s = s0 * (0.98 + 0.04 * rng.random())  # spacing slightly below target
        ox = rng.uniform(-0.5, 0.5)
        oy = rng.uniform(-0.5, 0.5)
        cands = _hex_points(s, ox, oy)
        if cands.shape[0] < N:
            continue
        # r_min: we want to keep points that can fit; use a fraction of s
        r_min = 0.5 * s * 0.9
        sel = _greedy_select(cands, N, r_min)
        if sel.shape[0] < N:
            continue
        v = _score(sel)
        if v > best_v:
            best_v = v
            best_c = sel.copy()

    # polish the best
    if best_c is not None:
        if time.time() < DEADLINE - 20:
            polished, pv = _polish(best_c, iters=400)
            if pv > best_v:
                best_c = polished
                best_v = pv

    # final validity clamp: ensure inside square
    if best_c is None:
        # fallback: grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
    else:
        # clamp to square (should already be inside)
        best_c = np.clip(best_c, -HALF, HALF)

    return best_c
