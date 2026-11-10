"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 130
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pts / n


def _fibonacci(n, offset=0.0):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + offset)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _covering_radius(pts):
    """Compute covering radius in radians via convex-hull facet circumradii."""
    pts = _normalize(pts)
    # Use Delaunay on the sphere via 3D convex hull.
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: brute force max of nearest-neighbor based estimate
        return _covering_radius_brute(pts)
    simplices = hull.simplices  # (F,3) indices
    max_r = 0.0
    for tri in simplices:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        # circumcenter of spherical triangle = direction of (a x b) x (b x c) normalized
        # pole direction
        v = np.cross(np.cross(a, b), np.cross(b, c))
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v = v / nv
        # circumradius = max angle from v to a,b,c
        d = np.max(np.arccos(np.clip(np.array([np.dot(v, a), np.dot(v, b), np.dot(v, c)]), -1, 1)))
        if d > max_r:
            max_r = d
    return max_r


def _covering_radius_brute(pts, n_samples=20000):
    """Fallback: sample sphere, compute max nearest distance."""
    rng = np.random.default_rng(0)
    # Fibonacci sample of test points
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # nearest neighbor distance
    D = 1.0 - Y @ pts.T  # 1 - cos
    mincos = np.min(pts.T @ Y.T, axis=0) if False else np.max(pts.T @ Y.T, axis=0)
    # pts.T @ Y.T gives (N, n_samples); max over N is nearest
    mincos = np.max(pts.T @ Y.T, axis=0)
    return float(np.max(np.arccos(np.clip(mincos, -1, 1))))


def _energy_packing(pts, s=10.0):
    """Riesz s-energy; high s emphasizes close pairs (packing)."""
    pts = _normalize(pts)
    D = 1.0 - pts @ pts.T
    D = np.maximum(D, 1e-12)
    np.fill_diagonal(D, np.inf)
    return float(np.sum(D ** (-s / 2.0)))


def _min_pairwise(pts):
    pts = _normalize(pts)
    D = 1.0 - pts @ pts.T
    np.fill_diagonal(D, 2.0)
    return float(np.arccos(np.clip(1.0 - np.min(D), -1, 1)))


def _improve_covering(pts, max_iters=2000, lr=0.05, deadline=None):
    """Direct local optimization of covering radius via pole-based moves."""
    pts = _normalize(pts).copy()
    best = _covering_radius(pts)
    best_pts = pts.copy()
    rng = np.random.default_rng(1)
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Find the worst facet and its pole
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        simplices = hull.simplices
        worst_r = 0.0
        worst_tri = None
        worst_pole = None
        for tri in simplices:
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            v = np.cross(np.cross(a, b), np.cross(b, c))
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                continue
            v = v / nv
            d = np.max(np.arccos(np.clip(np.array([np.dot(v, a), np.dot(v, b), np.dot(v, c)]), -1, 1)))
            if d > worst_r:
                worst_r = d
                worst_tri = tri
                worst_pole = v
        if worst_tri is None:
            break
        # Move the three vertices toward the pole a bit to shrink the facet
        for idx in worst_tri:
            a = pts[idx]
            # gradient step: move a toward pole
            new_a = a + lr * (worst_pole - a)
            new_a = new_a / np.linalg.norm(new_a)
            pts[idx] = new_a
        pts = _normalize(pts)
        cur = _covering_radius(pts)
        if cur < best - 1e-12:
            best = cur
            best_pts = pts.copy()
        else:
            # revert and try a small random perturbation of the worst vertices
            pts = best_pts.copy()
            for idx in worst_tri:
                noise = rng.normal(0, 0.01, 3)
                new_a = pts[idx] + noise
                new_a = new_a / np.linalg.norm(new_a)
                pts[idx] = new_a
            pts = _normalize(pts)
            cur = _covering_radius(pts)
            if cur < best - 1e-12:
                best = cur
                best_pts = pts.copy()
            else:
                pts = best_pts.copy()
                # decay lr
                lr *= 0.95
                if lr < 1e-4:
                    break
    return best_pts, best


def _optimize_packing(pts, max_iters=3000, lr=0.01, deadline=None):
    """Maximize minimum pairwise distance via repulsive force."""
    pts = _normalize(pts).copy()
    rng = np.random.default_rng(2)
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Repulsive force: for each point, push away from close neighbors
        D = 1.0 - pts @ pts.T
        np.fill_diagonal(D, 2.0)
        # force magnitude ~ 1/(d^2) for d < threshold
        thresh = 0.5
        mask = D < thresh
        force = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if mask[i, j] and i != j:
                    d2 = max(D[i, j], 1e-6)
                    force[i] += (pts[i] - pts[j]) / d2
        # normalize force
        fn = np.linalg.norm(force, axis=1, keepdims=True)
        fn[fn == 0] = 1.0
        force = force / fn
        new_pts = pts + lr * force
        new_pts = _normalize(new_pts)
        # accept if min distance improved
        if _min_pairwise(new_pts) > _min_pairwise(pts) - 1e-12:
            pts = new_pts
        else:
            # random perturbation
            pts = _normalize(pts + rng.normal(0, 0.005, (N, 3)))
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Start from Fibonacci spiral
    pts = _fibonacci(N)

    # Phase 1: packing optimization (improves min distance, helps covering)
    remaining = deadline - time.time()
    if remaining > 20:
        pts = _optimize_packing(pts, max_iters=2000, lr=0.01, deadline=deadline - 20)

    # Phase 2: direct covering radius minimization
    remaining = deadline - time.time()
    if remaining > 5:
        pts, best = _improve_covering(pts, max_iters=5000, lr=0.05, deadline=deadline - 2)

    # Final normalization and validation
    pts = _normalize(pts)
    # Ensure no coincident points
    D = 1.0 - pts @ pts.T
    np.fill_diagonal(D, 2.0)
    if np.min(D) < 1e-6:
        # perturb
        rng = np.random.default_rng(3)
        pts = _normalize(pts + rng.normal(0, 1e-4, (N, 3)))
        pts = _normalize(pts)

    # Ensure not in a closed hemisphere: check that origin is inside convex hull
    # (a good covering won't be in a hemisphere, but enforce)
    try:
        hull = ConvexHull(pts)
        # origin inside hull if all facet normals point away from origin consistently
        # Simple check: sum of signed volumes
        simplices = hull.simplices
        inside = True
        for tri in simplices:
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            n = np.cross(b - a, c - a)
            if np.dot(n, a) < 0:
                inside = False
                break
        if not inside:
            # reflect some points to spread
            rng = np.random.default_rng(4)
            for _ in range(10):
                idx = rng.integers(N)
                pts[idx] = -pts[idx]
                pts = _normalize(pts)
                try:
                    hull2 = ConvexHull(pts)
                    simplices2 = hull2.simplices
                    inside2 = True
                    for tri in simplices2:
                        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
                        n = np.cross(b - a, c - a)
                        if np.dot(n, a) < 0:
                            inside2 = False
                            break
                    if inside2:
                        break
                except Exception:
                    pass
    except Exception:
        pass

    return _normalize(pts)
# EVOLVE-BLOCK-END
