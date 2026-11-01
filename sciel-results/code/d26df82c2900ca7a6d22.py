"""Stronger packer: structured init + force-based refinement + smooth-min objective."""
import numpy as np
import time

N = 50
_DEADLINE = 110.0  # safety margin under the 150 s kill
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _valid(points, r):
    if points is None or points.shape != (N, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if r <= 0 or not np.isfinite(r):
        return False
    d = np.linalg.norm(points, axis=1)
    if np.any(d + r > 1.0 + 1e-9):
        return False
    # pairwise
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(points[i] - points[j]) < 2.0 * r - 1e-9:
                return False
    return True


def _score(points):
    d = np.linalg.norm(points, axis=1)
    edge = 1.0 - d
    r_edge = np.min(edge)
    r_pair = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            v = np.linalg.norm(points[i] - points[j]) / 2.0
            if v < r_pair:
                r_pair = v
    return min(r_edge, r_pair)


def _hex_init():
    """Hexagonal lattice cut to the disk, pick best N."""
    best = None
    for spacing in np.linspace(0.10, 0.20, 60):
        k = int(1.0 / spacing) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[rad <= 1.0 - spacing / 2.0]
        if len(pts) >= N:
            pts = pts[np.argsort(rad, kind="stable")[:N]]
            s = _score(pts)
            if best is None or s > best[0]:
                best = (s, pts.copy())
    if best is None:
        # fallback grid
        pts = np.zeros((N, 2))
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts[:, 0] = 0.5 * np.cos(ang)
        pts[:, 1] = 0.5 * np.sin(ang)
        return pts
    return best[1]


def _ring_init():
    """Concentric ring structure (known good for circle packing)."""
    # rings: 1 center + rings of increasing count
    # For 50: 1, 6, 12, 18, 13 -> 50
    ring_counts = [1, 6, 12, 18, 13]
    pts = []
    radii = [0.0, 0.28, 0.52, 0.72, 0.92]
    for cnt, rad in zip(ring_counts, radii):
        if cnt == 1:
            pts.append([0.0, 0.0])
        else:
            ang = np.linspace(0, 2 * np.pi, cnt, endpoint=False)
            pts.append(np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1))
    pts = np.vstack(pts)
    return pts[:N]


def _init():
    cands = [_hex_init(), _ring_init()]
    best = None
    for c in cands:
        if c.shape == (N, 2) and np.all(np.isfinite(c)):
            s = _score(c)
            if best is None or s > best[0]:
                best = (s, c.copy())
    if best is None:
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        c = np.stack([0.5 * np.cos(ang), 0.5 * np.sin(ang)], axis=1)
        return c
    return best[1]


def _soft_min(vals, t):
    """Soft-min surrogate via log-sum-exp (lower t = sharper min)."""
    vals = np.asarray(vals, dtype=float)
    m = np.min(vals)
    return m - (1.0 / t) * np.log(np.sum(np.exp(-t * (vals - m))))


def _objective_and_grad(points, t):
    """Return (smoothed objective value, gradient) where objective is -r (minimize).
    r = min(edge_margins, pair_margins). We minimize -r (i.e. maximize r)."""
    d = np.linalg.norm(points, axis=1)
    edge = 1.0 - d  # N values
    pair = []
    for i in range(N):
        for j in range(i + 1, N):
            pair.append(np.linalg.norm(points[i] - points[j]) / 2.0)
    pair = np.array(pair)
    vals = np.concatenate([edge, pair])
    # gradient of soft-min of vals wrt points
    w = np.exp(-t * (vals - np.min(vals)))
    w = w / np.sum(w)
    grad = np.zeros_like(points)
    # edge contribution
    for i in range(N):
        gi = w[i]
        # d edge_i / d points[i] = -points[i]/|points[i]|
        di = np.linalg.norm(points[i])
        if di > 1e-12:
            grad[i] -= gi * points[i] / di
    # pair contribution
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            wi = w[N + idx]
            diff = points[i] - points[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                # d(pair_ij)/d points[i] = (points[i]-points[j])/(2*dist)
                u = diff / dist
                grad[i] += wi * u / 2.0
                grad[j] -= wi * u / 2.0
            idx += 1
    val = _soft_min(vals, t)
    return -val, -grad  # minimize -r


def _refine(points, max_iter=4000):
    """Gradient descent on smoothed objective with adaptive step."""
    pts = points.copy()
    t = 12.0
    step = 0.02
    best_pts = pts.copy()
    best_val = -_score(pts)
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 5.0:
            break
        # anneal sharpness
        t = max(8.0, 12.0 * (0.9995 ** it))
        val, grad = _objective_and_grad(pts, t)
        if val < best_val:
            best_val = val
            best_pts = pts.copy()
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-8:
            break
        # normalize-ish step
        lr = step / (1.0 + gnorm)
        new = pts - lr * grad
        # keep inside-ish
        if _elapsed() > _DEADLINE - 5.0:
            break
        # accept if improved on true score
        if _score(new) > _score(pts):
            pts = new
        else:
            step *= 0.9
            if step < 1e-5:
                break
    return best_pts


def _polish(points, max_iter=2000):
    """High-precision local polish: coordinate-descent / steepest on true min."""
    pts = points.copy()
    step = 0.005
    best = _score(pts)
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 3.0:
            break
        improved = False
        # find the binding constraint (min of margins)
        d = np.linalg.norm(pts, axis=1)
        edge = 1.0 - d
        r_edge = np.min(edge)
        r_pair = np.inf
        bi, bj = -1, -1
        for i in range(N):
            for j in range(i + 1, N):
                v = np.linalg.norm(pts[i] - pts[j]) / 2.0
                if v < r_pair:
                    r_pair = v
                    bi, bj = i, j
        r = min(r_edge, r_pair)
        # move the most-constrained point(s) outward
        if r_edge <= r_pair:
            # push the point closest to boundary toward center
            i = int(np.argmin(edge))
            di = np.linalg.norm(pts[i])
            if di > 1e-12:
                dir = -pts[i] / di
            else:
                dir = np.array([0.0, 0.0])
            cand = pts.copy()
            cand[i] = pts[i] + step * dir
            if _score(cand) > best + 1e-12:
                pts = cand
                best = _score(pts)
                improved = True
        else:
            # separate the closest pair
            diff = pts[bi] - pts[bj]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                u = diff / dist
            else:
                u = np.array([1.0, 0.0])
            cand = pts.copy()
            cand[bi] = pts[bi] + 0.5 * step * u
            cand[bj] = pts[bj] - 0.5 * step * u
            if _score(cand) > best + 1e-12:
                pts = cand
                best = _score(pts)
                improved = True
        if not improved:
            step *= 0.8
            if step < 1e-7:
                break
    return pts


def construct():
    global _T0
    _T0 = time.time()
    pts = _init()
    pts = _refine(pts)
    pts = _polish(pts)
    # ensure validity; if not, fall back
    r = _score(pts)
    if not _valid(pts, r):
        pts = _init()
    return pts
