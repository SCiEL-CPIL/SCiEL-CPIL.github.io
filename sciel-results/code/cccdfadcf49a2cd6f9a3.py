"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _fibonacci_spiral(N):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere_points(N, rng):
    """Generate N random points on the unit sphere."""
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _perturbed_fibonacci(N, rng, noise_scale=0.1):
    """Generate a perturbed Fibonacci spiral."""
    base = _fibonacci_spiral(N)
    noise = rng.randn(N, 3) * noise_scale
    pts = base + noise
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _make_sample_grid(n=400):
    """A fixed, dense, well-spread set of probe points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _cover_radius_fast(pts, sample):
    """
    Fast vectorized estimate of the covering radius (radians).
    Max over probe points of the angular distance to the nearest configuration
    point. Cheap (no ConvexHull) so it can be called many times per second.
    """
    # Nearest dot product (max) for each probe point and for each config point.
    d_sample = sample @ pts.T          # (M, N)
    max_dot_sample = d_sample.max(axis=1)
    d_pts = pts @ pts.T                # (N, N)
    max_dot_pts = d_pts.max(axis=1)

    all_max = np.concatenate([max_dot_sample, max_dot_pts])
    all_max = np.clip(all_max, -1.0, 1.0)
    min_max = all_max.min()            # smallest of the per-point max dot products
    return float(np.arccos(min_max))


def _spherical_circumcenter(a, b, c):
    """
    Circumcenter of a spherical triangle on the unit sphere: the point p on the
    sphere with p·a = p·b = p·c. p is parallel to (a-b) x (a-c). We return the
    sign that maximizes the angular distance to the nearest vertex (the 'empty-cap'
    / covering-relevant circumcenter).
    """
    n = np.cross(a - b, a - c)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-12:
        centroid = a + b + c
        norm_c = np.linalg.norm(centroid)
        if norm_c < 1e-12:
            return a
        return centroid / norm_c

    p1 = n / norm_n
    p2 = -p1

    min_dot1 = min(np.dot(p1, a), np.dot(p1, b), np.dot(p1, c))
    min_dot2 = min(np.dot(p2, a), np.dot(p2, b), np.dot(p2, c))
    # Smaller min dot product => larger angular distance => the covering circumcenter.
    if min_dot1 <= min_dot2:
        return p1
    return p2


def _facet_circumcenters(pts, hull):
    """Circumcenter and angular circumradius for each triangular facet."""
    n_facets = len(hull.simplices)
    ccs = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    for i, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        cc = _spherical_circumcenter(a, b, c)
        ccs[i] = cc
        dots = np.array([np.dot(cc, a), np.dot(cc, b), np.dot(cc, c)])
        min_dot = float(np.clip(np.min(dots), -1.0, 1.0))
        radii[i] = np.arccos(min_dot)
    return ccs, radii


def _cover_radius_exact(pts):
    """
    Exact covering radius (radians) = max circumradius over the convex-hull
    facets, each checked to have an empty cap (no other point beyond the facet
    plane). Used only for final certification.
    """
    if len(pts) < 3:
        return np.pi
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fall back to a dense estimate if the hull is degenerate.
        return _cover_radius_fast(pts, _make_sample_grid(800))

    ccs, radii = _facet_circumcenters(pts, hull)
    max_r = float(np.max(radii))

    # Empty-cap check: a facet's circumcenter only governs the covering radius if
    # no other configuration point lies strictly beyond the facet plane.
    keep = np.ones(len(hull.simplices), dtype=bool)
    for i, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        normal = np.cross(b - a, c - a)
        nn = np.linalg.norm(normal)
        if nn < 1e-14:
            continue
        normal = normal / nn
        d = np.dot(normal, a)
        # Any point whose projection is beyond the facet plane (toward the origin
        # side of the outward normal) invalidates this cap.
        beyond = np.dot(pts, normal) < d - 1e-12
        if np.any(beyond):
            keep[i] = False

    if np.any(keep):
        return float(np.max(radii[keep]))
    # If every facet is occluded (should not happen for a proper hull), use the
    # max circumradius as a safe upper bound.
    return max_r


def _hill_climb_optimize(pts, sample, max_iters=20000, initial_step=0.02, time_budget=60.0):
    """
    Hill climbing: repeatedly identify the worst facet (via exact hull +
    circumcenters) and move one of its vertices away from that facet's
    circumcenter, accepting only if the fast covering radius decreases.
    """
    pts = pts.copy()
    step = initial_step
    best_radius = _cover_radius_fast(pts, sample)
    no_improve = 0
    start_time = time.time()

    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break

        try:
            hull = ConvexHull(pts)
        except Exception:
            break

        ccs, radii = _facet_circumcenters(pts, hull)
        max_idx = int(np.argmax(radii))
        critical = hull.simplices[max_idx]
        cc = ccs[max_idx]

        improved = False
        # Try moving each vertex of the critical facet away from the circumcenter.
        for idx in critical:
            v = pts[idx]
            dir_vec = v - cc
            dn = np.linalg.norm(dir_vec)
            if dn < 1e-15:
                continue
            dir_vec = dir_vec / dn

            new_pts = pts.copy()
            np_ = v + step * dir_vec
            np_ = np_ / np.linalg.norm(np_)
            new_pts[idx] = np_

            new_radius = _cover_radius_fast(new_pts, sample)
            if new_radius < best_radius - 1e-12:
                pts = new_pts
                best_radius = new_radius
                improved = True
                break

        if not improved:
            step *= 0.85
            if step < 1e-9:
                break
            no_improve += 1
            if no_improve > 60:
                break
        else:
            no_improve = 0
            if it % 40 == 0 and step < initial_step:
                step = min(step * 1.05, initial_step)

    return pts, best_radius


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before the 150s wall-clock kill.

    sample = _make_sample_grid(500)

    # Initial configuration: Fibonacci spiral.
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_fast = _cover_radius_fast(best_pts, sample)

    rng = np.random.RandomState(42)

    # Build a diverse set of starting configurations.
    starts = [pts.copy()]
    for k in range(4):
        starts.append(_perturbed_fibonacci(N, rng, noise_scale=0.05 + 0.03 * k))
    for k in range(3):
        starts.append(_random_sphere_points(N, rng))
    for k in range(1, 4):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i + k * 0.37
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        starts.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

    # Optimize each start with a fair share of the remaining budget.
    for s in starts:
        if time.time() > deadline - 25:
            break
        remaining = deadline - 25 - (time.time() - start_time)
        if remaining < 8:
            break
        budget = min(remaining * 0.6, 22.0)
        if budget < 4:
            continue
        opt, opt_fast = _hill_climb_optimize(s, sample, max_iters=20000,
                                             initial_step=0.02, time_budget=budget)
        if opt_fast < best_fast:
            best_fast = opt_fast
            best_pts = opt

    # Final refinement on the best configuration with a smaller step.
    if time.time() < deadline - 8:
        budget = deadline - 8 - (time.time() - start_time)
        if budget > 5:
            opt, opt_fast = _hill_climb_optimize(best_pts, sample, max_iters=30000,
                                                 initial_step=0.005, time_budget=budget)
            if opt_fast < best_fast:
                best_fast = opt_fast
                best_pts = opt

    # Final exact certification / validity handling.
    best_pts = best_pts.copy()
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)

    # Guard against configurations confined to a closed hemisphere (invalid).
    # A valid 50-point covering must span the sphere; check that the points are
    # not all within any closed hemisphere by verifying the origin is inside
    # their convex hull (equivalent to not lying in a closed hemisphere).
    try:
        hull = ConvexHull(best_pts)
    except Exception:
        hull = None
    if hull is None:
        best_pts = _fibonacci_spiral(N)
    else:
        # Origin inside convex hull <=> not in a closed hemisphere.
        try:
            from scipy.spatial import QhullError  # noqa: F401
        except Exception:
            pass
        # Simple hemisphere test: if there exists a direction u with u·p >= 0
        # for all p, the set lies in a closed hemisphere. For a full-rank
        # convex hull of a sphere covering, the origin is interior. We check
        # whether the origin is a convex combination (interior point).
        A = best_pts
        # Solve A^T w = 0 with w >= 0, sum w = 1, w > 0 (interior).
        # Use linear programming-free check: origin in interior of convex hull
        # iff for every direction u, min_p u·p < 0 < max_p u·p.
        # Sample directions to verify.
        rng2 = np.random.RandomState(7)
        u = rng2.randn(200, 3)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        proj = A @ u.T  # (N, 200)
        ok = True
        for j in range(u.shape[0]):
            col = proj[:, j]
            if col.min() >= -1e-9 or col.max() <= 1e-9:
                ok = False
                break
        if not ok:
            best_pts = _fibonacci_spiral(N)

    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
# EVOLVE-BLOCK-END
