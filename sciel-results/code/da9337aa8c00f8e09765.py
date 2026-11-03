"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from itertools import combinations

N = 21
DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def _validate(circles):
    """Check disjoint interiors and bounding box perimeter <= 4. Return (valid, perimeter, sum_r)."""
    if circles is None or len(circles) != N:
        return False, 9e9, 0.0
    xs = circles[:, 0]; ys = circles[:, 1]; rs = circles[:, 2]
    if np.any(rs <= 0):
        return False, 9e9, 0.0
    xmin, xmax = xs.min() - rs.min(), xs.max() + rs.max()
    ymin, ymax = ys.min() - rs.min(), ys.max() + rs.max()
    w, h = xmax - xmin, ymax - ymin
    per = 2 * (w + h)
    if per > 4.0 + 1e-9:
        return False, per, 0.0
    # pairwise non-overlap (disjoint interiors: dist >= r_i + r_j)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
            if d < rs[i] + rs[j] - 1e-9:
                return False, per, 0.0
    return True, per, rs.sum()


def _score(circles):
    """Objective = sum of radii if valid, else -inf."""
    ok, per, s = _validate(circles)
    return s if ok else -np.inf


def _grid_init(aspect):
    """Grid-like initial packing for a given aspect ratio (w/h)."""
    # perimeter 4 => w + h = 2. aspect = w/h => h = 2/(1+aspect), w = 2*aspect/(1+aspect)
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)
    # choose grid columns/rows to approximate N cells
    best = None
    for cols in range(1, 8):
        rows = int(np.ceil(N / cols))
        if rows < 1:
            continue
        if cols * rows < N:
            continue
        # cell size
        cw, ch = w / cols, h / rows
        r0 = 0.5 * min(cw, ch) * 0.98
        pts = []
        for i in range(cols):
            for j in range(rows):
                if len(pts) >= N:
                    break
                cx = (i + 0.5) * cw
                cy = (j + 0.5) * ch
                pts.append([cx, cy, r0])
        if len(pts) < N:
            continue
        pts = np.array(pts[:N])
        ok, per, s = _validate(pts)
        if ok and (best is None or s > best[0]):
            best = (s, pts.copy())
    if best is None:
        # fallback: tiny circles in a line
        r = 0.05
        pts = np.array([[i * (2 * r + 0.01), 0.5, r] for i in range(N)])
        return pts
    return best[1]


def _aspect_candidates():
    # explore a spread of aspect ratios; w+h=2 fixed
    return np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0])


def _anneal(circles, T0, T1, steps, seed):
    """Simulated annealing to grow radii and repack. Returns best valid circles."""
    rng = np.random.default_rng(seed)
    c = circles.copy()
    best = c.copy()
    best_s = _score(c)
    n = len(c)
    # precompute pairs
    pairs = list(combinations(range(n), 2))
    cur_s = best_s
    for it in range(steps):
        T = T0 * (T1 / T0) ** (it / max(1, steps - 1))
        # pick a random circle, perturb position and radius
        i = rng.integers(n)
        dx = rng.normal(0, 0.05)
        dy = rng.normal(0, 0.05)
        dr = rng.normal(0, 0.03)
        old = c[i].copy()
        c[i, 0] += dx
        c[i, 1] += dy
        c[i, 2] = max(1e-4, c[i, 2] + dr)
        new_s = _score(c)
        # accept
        if new_s > cur_s:
            cur_s = new_s
            if new_s > best_s:
                best_s = new_s
                best = c.copy()
        else:
            # Metropolis with score difference (validity-aware)
            if cur_s > -1e18 and new_s > -1e18:
                delta = new_s - cur_s
                if delta < 0 and rng.random() < np.exp(delta / max(T, 1e-9)):
                    cur_s = new_s
                else:
                    c[i] = old
            else:
                # one or both invalid; accept with small prob to escape
                if rng.random() < 0.05:
                    cur_s = new_s
                else:
                    c[i] = old
    return best, best_s


def _polish(circles, iters=200, seed=0):
    """Local polish: greedy radius growth + position nudge, keep valid."""
    rng = np.random.default_rng(seed)
    c = circles.copy()
    s = _score(c)
    if not np.isfinite(s):
        return c
    n = len(c)
    for it in range(iters):
        # try to grow a random radius
        i = rng.integers(n)
        old = c[i].copy()
        c[i, 2] *= 1.02
        ns = _score(c)
        if np.isfinite(ns) and ns > s:
            s = ns
        else:
            c[i] = old
            # try small position nudge
            c[i, 0] += rng.normal(0, 0.02)
            c[i, 1] += rng.normal(0, 0.02)
            ns = _score(c)
            if np.isfinite(ns) and ns > s:
                s = ns
            else:
                c[i] = old
    return c


def construct():
    t0 = time.time()
    best_c = None
    best_s = -np.inf

    aspects = _aspect_candidates()
    # sort aspects to try near-square first (often good for packing)
    aspects = aspects[np.argsort(np.abs(aspects - 1.0))]

    seed = 0
    # Phase 1: initialize and anneal each aspect
    for aspect in aspects:
        if time.time() > DEADLINE - 30:
            break
        c0 = _grid_init(aspect)
        s0 = _score(c0)
        if np.isfinite(s0):
            if s0 > best_s:
                best_s = s0
                best_c = c0.copy()
        # anneal with decreasing temperature
        T0, T1, steps = 0.5, 0.01, 400
        c, s = _anneal(c0, T0, T1, steps, seed)
        if np.isfinite(s) and s > best_s:
            best_s = s
            best_c = c.copy()
        seed += 1

    # Phase 2: polish the best
    if best_c is not None and np.isfinite(best_s):
        if time.time() < DEADLINE - 10:
            best_c = _polish(best_c, iters=300, seed=12345)
            s = _score(best_c)
            if np.isfinite(s):
                best_s = s

    # Phase 3: a few extra restarts from best with small perturbations if time
    if best_c is not None and np.isfinite(best_s):
        for k in range(3):
            if time.time() > DEADLINE - 15:
                break
            c = best_c.copy()
            rng = np.random.default_rng(1000 + k)
            # perturb positions slightly
            c[:, 0] += rng.normal(0, 0.03, N)
            c[:, 1] += rng.normal(0, 0.03, N)
            c, s = _anneal(c, 0.3, 0.01, 300, 5000 + k)
            if np.isfinite(s) and s > best_s:
                best_s = s
                best_c = c.copy()
            if time.time() < DEADLINE - 10:
                best_c = _polish(best_c, iters=150, seed=20000 + k)
                s = _score(best_c)
                if np.isfinite(s):
                    best_s = s

    if best_c is None or not np.isfinite(best_s):
        # absolute fallback: valid tiny packing
        r = 0.02
        best_c = np.array([[i * (2 * r + 0.02), 0.5, r] for i in range(N)])

    return best_c
