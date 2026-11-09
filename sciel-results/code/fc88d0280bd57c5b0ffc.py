"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 300


def _compute_score(points):
    """Packing radius: min(min_pairwise_dist/2, 1 - max_norm)."""
    n = len(points)
    if n < 2:
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    min_dist = d.min()
    norms = np.hypot(points[:, 0], points[:, 1])
    return float(min(min_dist / 2.0, 1.0 - norms.max()))


def _hex_lattice_init(n_points):
    """Best hexagonal lattice packing of n_points inside the unit disk."""
    best_pts = None
    best_score = -1.0
    for scale in np.linspace(0.088, 0.145, 70):
        dx = scale
        dy = scale * np.sqrt(3.0) / 2.0
        n_rows = int(2.0 / dy) + 2
        n_cols = int(2.0 / dx) + 2
        pts = []
        for i in range(-n_cols, n_cols + 1):
            for j in range(-n_rows, n_rows + 1):
                x = i * dx + (0.5 * dx if (j % 2) else 0.0)
                y = j * dy
                if x * x + y * y <= 1.0:
                    pts.append((x, y))
        pts = np.array(pts, dtype=float)
        if len(pts) < n_points:
            continue
        radii = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(radii, kind="stable")[:n_points]
        cand = pts[idx]
        s = _compute_score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand
    if best_pts is None:
        best_pts = np.random.randn(n_points, 2) * 0.5
        best_score = 0.0
    return best_pts, best_score


def _refine(pts, deadline):
    """Local refinement maximizing a repulsive-potential objective (proxy for min dist)."""
    n = len(pts)
    pts = pts.astype(float).copy()

    def objective(x):
        p = x.reshape(n, 2)
        diff = p[:, None, :] - p[None, :, :]
        d2 = (diff ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)
        norms = np.hypot(p[:, 0], p[:, 1])
        r = 1.0 - norms.max()
        return float(np.mean(1.0 / (d + 1e-6)) + 1e6 * r * r)

    x = pts.reshape(-1)
    obj = objective(x)
    # Vectorized steepest descent with backtracking line search.
    it = 0
    while time.time() < deadline and it < 4000:
        it += 1
        p = x.reshape(n, 2)
        diff = p[:, None, :] - p[None, :, :]
        d2 = (diff ** 2).sum(-1)
        np.fill_diagonal(d2, 1e-12)
        d = np.sqrt(d2)
        inv = 1.0 / (d + 1e-6)
        # grad of mean(1/d): -mean_j (1/d_ij^2) * (p_i - p_j)/d_ij
        g = -np.einsum('ij,ijk->ik', inv * inv, diff) / n
        norms = np.hypot(p[:, 0], p[:, 1])
        i_max = int(np.argmax(norms))
        if norms[i_max] > 1e-9:
            g[i_max] += 2.0e6 * (1.0 - norms[i_max]) * p[i_max] / norms[i_max]
        g = g.reshape(-1)
        gn = np.linalg.norm(g)
        if gn < 1e-13:
            break
        g = g / gn
        # backtracking line search
        step = 0.05
        for _ in range(15):
            x2 = x - step * g
            if objective(x2) < obj:
                x = x2
                obj = objective(x)
                break
            step *= 0.5
        else:
            break
    return x.reshape(n, 2)


def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin before the 150s kill

    pts, init_r = _hex_lattice_init(N)
    best_pts = pts.copy()
    best_r = init_r

    # Refinement 1: deterministic steepest descent.
    if time.time() < deadline - 5.0:
        cand = _refine(pts, deadline - 5.0)
        r = _compute_score(cand)
        if r > best_r:
            best_r, best_pts = r, cand.copy()

    # Refinement 2: a few perturbed restarts from the best so far.
    rng = np.random.default_rng(12345)
    for _ in range(4):
        if time.time() >= deadline - 8.0:
            break
        jitter = best_pts + 0.004 * rng.standard_normal((N, 2))
        cand = _refine(jitter, deadline - 8.0)
        r = _compute_score(cand)
        if r > best_r:
            best_r, best_pts = r, cand.copy()

    # Final safety: if anything went wrong, fall back to a valid lattice.
    if best_pts.shape != (N, 2) or not np.all(np.isfinite(best_pts)):
        best_pts, _ = _hex_lattice_init(N)

    return best_pts
