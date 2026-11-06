"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 50

# EVOLVE-BLOCK-START
import time

_DEADLINE = time.time() + 110.0  # safety margin under the 150 s wall-clock kill


def _pairwise_min(pts):
    """Return the minimum pairwise distance and the (i, j) index pair achieving it."""
    n = len(pts)
    dx = pts[:, None, 0] - pts[None, :, 0]
    dy = pts[:, None, 1] - pts[None, :, 1]
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, np.inf)
    k = np.argmin(d2)
    i, j = np.unravel_index(k, (n, n))
    return np.sqrt(d2[i, j]), (i, j)


def _score(pts):
    """Largest common radius: min of half the closest-pair distance and the
    closest distance to the container boundary."""
    min_pair, _ = _pairwise_min(pts)
    margin = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair * 0.5, margin)


def _ring_seed(rng, n, jitter=0.0):
    """Concentric-ring seed: 1 centre + rings, hexagonally-ish spaced, then jittered."""
    pts = [np.array([0.0, 0.0])]
    # choose spacing so that ~n points fit inside radius 1
    # approximate: area of unit disk / area per point
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    ring_r = spacing
    ring_idx = 1
    while len(pts) < n and ring_r < 0.95:
        # number of points on this ring ~ circumference / spacing
        m = max(6, int(round(2.0 * np.pi * ring_r / spacing)))
        for k in range(m):
            if len(pts) >= n:
                break
            ang = 2.0 * np.pi * k / m + (ring_idx % 2) * (np.pi / m)
            pts.append(np.array([ring_r * np.cos(ang), ring_r * np.sin(ang)]))
        ring_r += spacing * 0.9
        ring_idx += 1
    # if still short, add random points
    while len(pts) < n:
        r = rng.random()
        a = rng.random() * 2 * np.pi
        pts.append(np.array([np.sqrt(r) * np.cos(a), np.sqrt(r) * np.sin(a)]))
    pts = np.array(pts[:n], dtype=float)
    if jitter > 0:
        pts += rng.normal(0, jitter, size=pts.shape)
        pts = np.clip(pts, -0.99, 0.99)
    return pts


def _repulsion_optimize(pts, rng, iters=4000, base_step=0.02):
    """Vectorized repulsion: push pairs apart, pull to boundary if outside.
    Returns improved pts."""
    pts = pts.copy()
    n = len(pts)
    for it in range(iters):
        # pairwise distances
        dx = pts[:, None, 0] - pts[None, :, 0]
        dy = pts[:, None, 1] - pts[None, :, 1]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        # repulsion force: for pairs closer than 2r_target (use current min as proxy)
        # use a soft repulsion: force ~ 1/d for all pairs, scaled
        # direction to push apart
        inv = 1.0 / d
        inv[np.diag_indices(n)] = 0.0
        fx = dx * inv
        fy = dy * inv
        # sum forces on each point (push away from neighbours)
        fx_sum = fx.sum(axis=1)
        fy_sum = fy.sum(axis=1)
        # boundary: if |p| > 1 - margin, pull inward
        r = np.hypot(pts[:, 0], pts[:, 1])
        # desired margin ~ current best radius estimate
        best = _score(pts)
        target = 1.0 - best
        over = r - target
        pull = np.where(over > 0, -over, 0.0)
        fx_sum += pull * pts[:, 0] / np.maximum(r, 1e-9)
        fy_sum += pull * pts[:, 1] / np.maximum(r, 1e-9)
        # adaptive step
        step = base_step * (1.0 / (1.0 + 0.01 * it))
        pts += step * np.column_stack([fx_sum, fy_sum])
        # clamp inside
        r = np.hypot(pts[:, 0], pts[:, 1])
        scale = np.minimum(1.0, (1.0 - 1e-4) / np.maximum(r, 1e-9))
        pts *= scale[:, None]
    return pts


def _newton_refine(pts, max_iters=60, tol=1e-9):
    """Newton-Raphson on active contact constraints:
    - pairs at minimum distance should be at 2r
    - circles at max radius should be at 1 - r
    Solve for r and positions so that the active constraints are exactly satisfied.
    Returns refined pts and the achieved r."""
    pts = pts.copy()
    n = len(pts)
    r = _score(pts)
    for _ in range(max_iters):
        # find active constraints
        min_pair, (i, j) = _pairwise_min(pts)
        r_i = np.hypot(pts[i, 0], pts[i, 1])
        r_j = np.hypot(pts[j, 0], pts[j, 1])
        boundary_margin = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
        # current limiting radius
        r_cur = min(min_pair * 0.5, boundary_margin)
        # build linear system: unknowns = positions (2n) and r (1)
        # constraints:
        # 1) |p_i - p_j| = 2r  (the closest pair)
        # 2) for each circle on boundary (|p_k| = 1 - r), |p_k| = 1 - r
        # We solve a least-squares on the active set.
        # Collect active pairs (those within 2r + tol) and active boundary circles.
        active_pairs = []
        active_boundary = []
        d = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                     pts[:, None, 1] - pts[None, :, 1])
        np.fill_diagonal(d, np.inf)
        for a in range(n):
            for b in range(a + 1, n):
                if d[a, b] <= 2.0 * r_cur + 1e-6:
                    active_pairs.append((a, b))
        for k in range(n):
            rk = np.hypot(pts[k, 0], pts[k, 1])
            if rk >= 1.0 - r_cur - 1e-6:
                active_boundary.append(k)
        if not active_pairs and not active_boundary:
            break
        # Build Jacobian: rows = constraints, cols = [dx_0, dy_0, ..., dx_{n-1}, dy_{n-1}, dr]
        rows = []
        rhs = []
        for (a, b) in active_pairs:
            dx = pts[a, 0] - pts[b, 0]
            dy = pts[a, 1] - pts[b, 1]
            d_ab = np.hypot(dx, dy)
            if d_ab < 1e-12:
                continue
            ux = dx / d_ab
            uy = dy / d_ab
            # constraint: |p_a - p_b| - 2r = 0
            # d/dp_a = (ux, uy), d/dp_b = (-ux, -uy), d/dr = -2
            row = np.zeros(2 * n + 1)
            row[2 * a] = ux
            row[2 * a + 1] = uy
            row[2 * b] = -ux
            row[2 * b + 1] = -uy
            row[2 * n] = -2.0
            rows.append(row)
            rhs.append(2.0 * r_cur - d_ab)  # we want |p_a-p_b| = 2r
        for k in active_boundary:
            rk = np.hypot(pts[k, 0], pts[k, 1])
            if rk < 1e-12:
                continue
            ux = pts[k, 0] / rk
            uy = pts[k, 1] / rk
            # constraint: |p_k| + r - 1 = 0  -> |p_k| = 1 - r
            row = np.zeros(2 * n + 1)
            row[2 * k] = ux
            row[2 * k + 1] = uy
            row[2 * n] = 1.0
            rows.append(row)
            rhs.append(1.0 - r_cur - rk)  # we want |p_k| = 1 - r
        if not rows:
            break
        A = np.array(rows)
        b = np.array(rhs)
        # solve least squares
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        # damp the step
        step = sol
        # limit step size to avoid divergence
        max_step = 0.05
        norm = np.linalg.norm(step[:2 * n])
        if norm > max_step:
            step[:2 * n] *= max_step / norm
        pts += step[:2 * n].reshape(n, 2)
        r_new = r_cur + step[2 * n]
        # clamp r to positive and reasonable
        r_new = np.clip(r_new, 0.0, 0.5)
        # clamp positions inside
        rr = np.hypot(pts[:, 0], pts[:, 1])
        scale = np.minimum(1.0, (1.0 - 1e-4) / np.maximum(rr, 1e-9))
        pts *= scale[:, None]
        r = r_new
        # check convergence
        if np.linalg.norm(step) < tol:
            break
    # final score
    return pts, _score(pts)


def construct():
    rng = np.random.default_rng(12345)
    best_pts = None
    best_score = -1.0

    # Try several seeds with varying jitter
    for attempt in range(20):
        if time.time() > _DEADLINE:
            break
        jitter = 0.0 if attempt == 0 else 0.02 * (1 + attempt * 0.1)
        pts = _ring_seed(rng, N, jitter=jitter)
        # repulsion optimization
        pts = _repulsion_optimize(pts, rng, iters=2000, base_step=0.015)
        # Newton refinement
        pts, sc = _newton_refine(pts, max_iters=40)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()

    if best_pts is None:
        # fallback: hexagonal lattice
        pts = _ring_seed(rng, N, jitter=0.0)
        best_pts = pts
        best_score = _score(pts)

    return best_pts
# EVOLVE-BLOCK-END
