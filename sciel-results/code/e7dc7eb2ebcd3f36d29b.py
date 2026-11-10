"""Ring-based packing for 300 equal circles in the unit circle."""
import time
import numpy as np

N = 300
_DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)


def _score(pts):
    """Compute the common radius for a set of centres."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # boundary
    r_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    if np.any(r_boundary < -1e-12):
        return 0.0
    # pairwise
    d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    r_pair = np.min(d) / 2.0
    return min(float(r_pair), float(np.min(r_boundary)))


def _ring_init(n_rings, per_ring):
    """Build a ring-based initial configuration.

    per_ring: list of circle counts per ring (inner to outer).
    """
    pts = []
    m = len(per_ring)
    if m == 0:
        return None
    # ring radii: distribute between 0 and ~0.97
    if m == 1:
        radii = [0.0]
    else:
        radii = np.linspace(0.0, 0.97, m)
    for k, (r, n) in enumerate(zip(radii, per_ring)):
        if n == 0:
            continue
        if n == 1:
            # put single point at ring radius (or at origin for innermost)
            pts.append([r, 0.0])
        else:
            theta = np.arange(n) * (2.0 * np.pi / n) + (0.0 if k % 2 == 0 else np.pi / n)
            pts.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    if not pts:
        return None
    arr = np.vstack(pts)
    return arr[:N] if len(arr) >= N else None


def _repulsion_refine(pts, iters, dt, rng):
    """Local optimization: repel overlapping pairs, pull boundary-violating points in."""
    pts = np.array(pts, dtype=np.float64, copy=True)
    n = len(pts)
    for it in range(iters):
        # pairwise distances
        d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(d, np.inf)
        # repulsion: push apart pairs closer than 2r (r estimated from current min)
        # use a soft repulsion based on current minimum distance
        r_cur = _score(pts)
        target = max(r_cur * 2.0, 1e-4)
        # force magnitude
        f = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < target:
                    diff = pts[i] - pts[j]
                    dist = d[i, j] if d[i, j] > 1e-12 else 1e-12
                    mag = (target - dist) / dist
                    f[i] += diff * mag
                    f[j] -= diff * mag
        # boundary constraint: pull points back inside
        norms = np.linalg.norm(pts, axis=1)
        for i in range(n):
            if norms[i] > 1.0:
                f[i] -= pts[i] / norms[i] * (norms[i] - 1.0)
        # small random jitter to escape local minima
        f += rng.normal(0, dt * 0.1, size=f.shape)
        pts += dt * f
        # hard clip to stay inside (with margin)
        norms = np.linalg.norm(pts, axis=1)
        for i in range(n):
            if norms[i] > 0.9999:
                pts[i] *= 0.9999 / norms[i]
    return pts


def _try_ring_config(per_ring, rng, max_refine_iters):
    """Build a ring config and refine it."""
    pts = _ring_init(len(per_ring), per_ring)
    if pts is None or len(pts) != N:
        return None
    base = _score(pts)
    # refine with decreasing step size
    cur = pts
    best = cur
    best_score = base
    for phase in range(4):
        iters = max_refine_iters // (phase + 1)
        dt = 0.005 / (phase + 1)
        cur = _repulsion_refine(cur, iters, dt, rng)
        s = _score(cur)
        if s > best_score:
            best_score = s
            best = cur.copy()
    return best, best_score


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)

    # Build candidate ring configurations: total must equal N
    candidates = []

    # Symmetric: k rings each with m circles, k*m = N
    # Factorizations of 300
    for k in range(1, 31):
        if N % k == 0:
            m = N // k
            per_ring = [m] * k
            candidates.append(per_ring)

    # Asymmetric: inner rings smaller
    # e.g., 1 center + rings
    for inner in range(0, 6):
        remaining = N - inner
        if remaining <= 0:
            continue
        # distribute remaining over rings
        for n_rings in range(1, 20):
            if remaining % n_rings == 0:
                m = remaining // n_rings
                per_ring = [1] * inner + [m] * n_rings
                if len(per_ring) <= 25:
                    candidates.append(per_ring)

    # Progressive: rings with increasing counts
    for base in range(5, 40):
        for growth in range(0, 5):
            per_ring = []
            total = 0
            for k in range(1, 30):
                cnt = base + growth * (k - 1)
                if total + cnt > N:
                    cnt = N - total
                if cnt <= 0:
                    break
                per_ring.append(cnt)
                total += cnt
                if total == N:
                    break
            if total == N and 1 <= len(per_ring) <= 25:
                candidates.append(per_ring)

    # Random ring configurations
    for _ in range(30):
        n_rings = rng.integers(3, 20)
        # random composition of N into n_rings parts
        cuts = sorted(rng.integers(1, N, size=n_rings - 1))
        per_ring = []
        prev = 0
        for c in cuts:
            per_ring.append(c - prev)
            prev = c
        per_ring.append(N - prev)
        per_ring = [x for x in per_ring if x > 0]
        if len(per_ring) == n_rings:
            candidates.append(per_ring)

    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    best_pts = None
    best_score = 0.0

    # Time budget per candidate
    total_time = _DEADLINE - (time.time() - t0)
    per_candidate = max(1, int(total_time / max(1, len(unique_candidates))))

    for idx, per_ring in enumerate(unique_candidates):
        elapsed = time.time() - t0
        if elapsed > _DEADLINE - 5:
            break
        # scale refine iters based on remaining time
        remaining = _DEADLINE - 5 - elapsed
        max_iters = int(remaining * 8)  # rough budget
        max_iters = max(20, min(max_iters, 400))
        result = _try_ring_config(per_ring, rng, max_iters)
        if result is not None:
            pts, sc = result
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
        # check deadline
        if time.time() - t0 > _DEADLINE - 10:
            break

    # Final polish on the best
    if best_pts is not None and time.time() - t0 < _DEADLINE - 10:
        remaining = _DEADLINE - 10 - (time.time() - t0)
        polish_iters = int(remaining * 5)
        polish_iters = max(20, min(polish_iters, 300))
        polished = _repulsion_refine(best_pts, polish_iters, 0.002, rng)
        s = _score(polished)
        if s > best_score:
            best_score = s
            best_pts = polished

    if best_pts is None:
        # Fallback: hexagonal lattice
        spacing = 0.1
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points = points[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_pts = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98

    return best_pts
